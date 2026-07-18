"""
gerar_dataset.py — destilacao: usa o Gemini como professor para gerar um dataset
sintetico de pares pergunta(portugues) -> Cypher, ancorado no esquema do grafo.

O dataset alimenta o fine-tuning QLoRA (notebooks/02_qlora_kaggle.ipynb): um modelo
pequeno e aberto aprende a traduzir portugues -> Cypher a partir das respostas do
professor. Reaproveita os padroes do pipeline AMR: rotacao de chaves, retry com
fallback de modelo, saida estruturada (JSON) e contabilidade de custo.

Uso:
    python src/gerar_dataset.py --n 400
    python src/gerar_dataset.py --n 400 --provider groq

Saida: data/processed/dataset_text2cypher.jsonl  (nao versionado)
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esquema import ESQUEMA_TEXTO, EXEMPLOS  # noqa: E402
from llm import LLM  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "data" / "processed" / "dataset_text2cypher.jsonl"

# Eixos de diversidade: forcam o professor a cobrir varios tipos de pergunta.
TEMAS = [
    "evasao (residentes que saem da regiao)",
    "invasao/atracao (polos que recebem de fora)",
    "indice de retencao regional (IRR) por complexidade",
    "comparacao entre linhas de cuidado (radioterapia, quimioterapia, diagnostico)",
    "distancia dos municipios ao polo e atributos socioeconomicos",
    "series por ano (2019 a 2024) e tendencia",
    "oferta: estabelecimentos e procedimentos que realizam",
    "ranking de regioes por volume ou por taxa de deslocamento",
]

PROMPT_PROFESSOR = """\
Voce e um especialista em Neo4j e no SUS. Gere {k} pares de pergunta em portugues
e a consulta Cypher correta para o grafo abaixo. As perguntas devem ser naturais,
como as de um gestor de saude, e variadas em fraseado. Foque especialmente no tema:
"{tema}".

Regras:
- Use SOMENTE rotulos, relacoes e propriedades do esquema. Nao invente campos.
- Cada Cypher deve ser sintaticamente valido e uma unica consulta.
- Varie nomes de regioes/municipios plausiveis de Sao Paulo.
- Nao repita perguntas ja listadas nos exemplos.

ESQUEMA:
{esquema}

EXEMPLOS (estilo alvo):
{exemplos}

Responda em JSON com a forma:
{{"pares": [{{"pergunta": "...", "cypher": "..."}}, ...]}}"""


def gerar(n, provider, por_chamada=8):
    llm = LLM(provider=provider)
    exemplos_txt = "\n".join(f"P: {e['pergunta']}\nC: {e['cypher']}" for e in EXEMPLOS)
    vistos = set()
    pares = []

    # semear com os exemplos curados
    for e in EXEMPLOS:
        chave = e["pergunta"].lower().strip()
        if chave not in vistos:
            vistos.add(chave)
            pares.append({"pergunta": e["pergunta"], "cypher": e["cypher"], "origem": "semente"})

    i = 0
    while len(pares) < n:
        tema = TEMAS[i % len(TEMAS)]
        i += 1
        k = min(por_chamada, n - len(pares) + 2)
        prompt = PROMPT_PROFESSOR.format(
            k=k, tema=tema, esquema=ESQUEMA_TEXTO, exemplos=exemplos_txt
        )
        try:
            obj = llm.gerar_json(prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  lote {i}: falhou ({str(e)[:80]}), seguindo")
            continue
        novos = 0
        for par in obj.get("pares", []):
            p = (par.get("pergunta") or "").strip()
            c = (par.get("cypher") or "").strip()
            if not p or not c:
                continue
            chave = p.lower()
            if chave in vistos:
                continue
            vistos.add(chave)
            pares.append({"pergunta": p, "cypher": c, "origem": "gemini"})
            novos += 1
        print(f"  lote {i} (tema: {tema[:32]}...): +{novos} pares | total {len(pares)}/{n}")

    return pares[:n], llm.custo()


def main():
    ap = argparse.ArgumentParser(description="Gera dataset sintetico pergunta->Cypher via LLM professor")
    ap.add_argument("--n", type=int, default=400, help="numero de pares a gerar")
    ap.add_argument("--provider", default="gemini", choices=["gemini", "groq"])
    ap.add_argument("--saida", default=str(SAIDA))
    args = ap.parse_args()

    print(f"Gerando {args.n} pares com provedor '{args.provider}'...")
    pares, custo = gerar(args.n, args.provider)

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        for par in pares:
            f.write(json.dumps(par, ensure_ascii=False) + "\n")

    print(f"\nSalvo {len(pares)} pares em {saida}")
    print(f"Custo (professor): {custo}")


if __name__ == "__main__":
    main()
