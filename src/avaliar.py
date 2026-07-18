"""
avaliar.py — avaliacao do Text2Cypher (parte de MLOps / avaliacao de modelos).

Para cada provedor (gemini, groq, local), gera o Cypher de cada pergunta do
conjunto eval/perguntas.yaml e mede duas coisas:
  1. Validade      — o Cypher gerado passa no EXPLAIN do Neo4j (sintaxe/schema).
  2. Acerto        — executado, produz o MESMO resultado que o Cypher de referencia.

Reporta, por provedor: % valido, % correto, tokens/custo. Escreve
outputs/tables/avaliacao.csv e imprime a tabela final (API vs modelo fine-tunado).

Uso:
    python src/avaliar.py --providers gemini,local
    python src/avaliar.py                # padrao: gemini
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import text2cypher as t2c  # noqa: E402

load_dotenv()
RAIZ = Path(__file__).resolve().parent.parent
PERGUNTAS = RAIZ / "eval" / "perguntas.yaml"
SAIDA = RAIZ / "outputs" / "tables" / "avaliacao.csv"


def _driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
    )


def cypher_valido(driver, cypher):
    """EXPLAIN nao executa, so valida sintaxe e schema."""
    try:
        with driver.session() as s:
            s.run(f"EXPLAIN {cypher}").consume()
        return True
    except Exception:  # noqa: BLE001
        return False


def _normaliza_linhas(linhas):
    """Conjunto de tuplas ordenadas, com floats arredondados, para comparar
    resultados independncia de ordem de colunas/linhas."""
    norm = set()
    for l in linhas:
        vals = []
        for v in l.values():
            vals.append(round(v, 4) if isinstance(v, float) else v)
        norm.add(tuple(sorted((str(k) for k in l.keys()))) + tuple(vals))
    return norm


def resultado(driver, cypher, limite=1000):
    with driver.session() as s:
        return [dict(r) for r in s.run(cypher)][:limite]


def gerar(provider, pergunta):
    if provider == "local":
        return t2c.gerar_local(pergunta)
    return t2c.gerar_api(pergunta, provider)


def avaliar_provider(provider, itens, driver):
    n = len(itens)
    validos = corretos = 0
    tokens = 0
    for it in itens:
        perg, ref = it["pergunta"], it["cypher"].strip()
        try:
            pred, custo = gerar(provider, perg)
        except Exception as e:  # noqa: BLE001
            print(f"  [{provider}] geracao falhou: {str(e)[:80]}")
            continue
        if custo:
            tokens += custo.get("total_tokens", 0)
        val = cypher_valido(driver, pred)
        validos += val
        ok = False
        if val:
            try:
                ok = _normaliza_linhas(resultado(driver, pred)) == _normaliza_linhas(resultado(driver, ref))
            except Exception:  # noqa: BLE001
                ok = False
        corretos += ok
        print(f"  [{provider}] {'OK' if ok else ('valido' if val else 'invalido')}: {perg[:55]}")
    return {
        "provedor": provider,
        "n": n,
        "validos": validos,
        "corretos": corretos,
        "pct_valido": round(100 * validos / n, 1),
        "pct_correto": round(100 * corretos / n, 1),
        "tokens_total": tokens,
    }


def main():
    ap = argparse.ArgumentParser(description="Avaliacao do Text2Cypher (validade + acerto)")
    ap.add_argument("--providers", default="gemini", help="lista separada por virgula: gemini,groq,local")
    args = ap.parse_args()
    provedores = [p.strip() for p in args.providers.split(",") if p.strip()]

    with open(PERGUNTAS, encoding="utf-8") as f:
        itens = yaml.safe_load(f)
    print(f"Conjunto: {len(itens)} perguntas | provedores: {', '.join(provedores)}")

    driver = _driver()
    try:
        driver.verify_connectivity()
    except Exception as e:  # noqa: BLE001
        print(f"Neo4j indisponivel: {e}\nSuba com 'docker compose up -d' e 'python src/grafo.py'.")
        return

    linhas = []
    try:
        for prov in provedores:
            print(f"\n== {prov} ==")
            linhas.append(avaliar_provider(prov, itens, driver))
    finally:
        driver.close()

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(SAIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    print("\n" + "=" * 60)
    print(f"{'provedor':<12} {'valido%':>8} {'correto%':>9} {'tokens':>10}")
    for l in linhas:
        print(f"{l['provedor']:<12} {l['pct_valido']:>8} {l['pct_correto']:>9} {l['tokens_total']:>10}")
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
