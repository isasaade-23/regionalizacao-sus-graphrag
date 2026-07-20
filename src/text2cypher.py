"""
text2cypher.py — GraphRAG: pergunta em portugues -> Cypher -> resposta do grafo.

Caminhos de geracao do Cypher:
  --provider gemini  (API, professor; padrao)
  --provider groq    (API, modelo aberto servido pela Groq)
  --provider mistral (API, modelo aberto da Mistral, ex.: codestral-latest)
  --provider ollama  (modelo aberto local via Ollama; sem chave nem rate limit)
  --provider local   (modelo fine-tunado com QLoRA: base + adapter LoRA)

O Cypher gerado e executado no Neo4j (config no .env). Use --no-run para so gerar.

Uso:
    python src/text2cypher.py "Quais regioes mais enviam pacientes de cancer para fora?"
    python src/text2cypher.py "..." --provider local
    python src/text2cypher.py "..." --no-run
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402
from esquema import montar_prompt  # noqa: E402

load_dotenv()


def limpar_cypher(txt):
    """Tira cercas de codigo e prefixos, deixa uma consulta."""
    t = re.sub(r"^```(?:cypher)?\s*", "", txt.strip())
    t = re.sub(r"\s*```$", "", t).strip()
    t = re.sub(r"^cypher\s*", "", t, flags=re.IGNORECASE).strip()
    return t.rstrip(";").strip()


# ── geracao via API (Gemini/Groq), reaproveitando o cliente LLM ──
def gerar_api(pergunta, provider):
    from llm import LLM

    llm = LLM(provider=provider)
    return limpar_cypher(llm.gerar(montar_prompt(pergunta))), llm.custo()


# ── geracao via modelo fine-tunado local (base + adapter LoRA) ──
_LOCAL = {}


def gerar_local(pergunta):
    if "modelo" not in _LOCAL:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = os.getenv("MODELO_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
        adapter = os.getenv("ADAPTER_DIR", "adapters/text2cypher-sus")
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        _LOCAL.update(tok=tok, modelo=model, torch=torch)

    tok, model, torch = _LOCAL["tok"], _LOCAL["modelo"], _LOCAL["torch"]
    prompt = montar_prompt(pergunta)
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=180, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return limpar_cypher(txt), None


# ── execucao no Neo4j ──
def executar(cypher, limite=25):
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        with driver.session() as sess:
            res = sess.run(cypher)
            linhas = [dict(r) for r in res][:limite]
        return linhas
    finally:
        driver.close()


def imprimir_tabela(linhas):
    if not linhas:
        print("(sem resultados)")
        return
    cols = list(linhas[0].keys())
    larg = {c: max(len(c), *(len(str(l.get(c, ""))) for l in linhas)) for c in cols}
    print(" | ".join(c.ljust(larg[c]) for c in cols))
    print("-+-".join("-" * larg[c] for c in cols))
    for l in linhas:
        print(" | ".join(str(l.get(c, "")).ljust(larg[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser(description="GraphRAG Text2Cypher em portugues sobre o grafo do SUS")
    ap.add_argument("pergunta", help="pergunta em portugues")
    ap.add_argument("--provider", default="gemini", choices=["gemini", "groq", "mistral", "ollama", "local"])
    ap.add_argument("--no-run", action="store_true", help="so gera o Cypher, nao executa")
    args = ap.parse_args()

    if args.provider == "local":
        cypher, custo = gerar_local(args.pergunta)
    else:
        cypher, custo = gerar_api(args.pergunta, args.provider)

    print(f"\nPergunta: {args.pergunta}")
    print(f"Cypher  : {cypher}")
    if custo:
        print(f"Custo   : {custo}")

    if not args.no_run:
        try:
            linhas = executar(cypher)
            print("\nResultado:")
            imprimir_tabela(linhas)
        except Exception as e:  # noqa: BLE001
            print(f"\nFalha ao executar no Neo4j: {e}")
            print("Suba o grafo com 'docker compose up -d' e 'python src/grafo.py', ou use --no-run.")


if __name__ == "__main__":
    main()
