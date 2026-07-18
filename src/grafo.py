"""
grafo.py — carrega a tabela de fluxo assistencial no Neo4j.

Le os parquets tidy de data/processed/ (produzidos por src/extracao.py) e monta o
grafo do esquema (schema/grafo.md): regioes, municipios (PERTENCE_A), fluxo
inter-regional (FLUXO) e, se presentes, estabelecimentos/procedimentos (REALIZA).

Contrato de dados (data/processed/):
  regioes.parquet        codigo, nome, drs, polo_oncologico
  municipios.parquet     codigo, nome, regiao_codigo, populacao, idhm_renda, dist_polo_km
  fluxo.parquet          reg_res_codigo, reg_res_nome, reg_aten_codigo, reg_aten_nome,
                         ano, complexidade, linha, volume, deslocou
  procedimentos.parquet  (opcional) codigo, nome, grupo, linha, complexidade, cid_grupo
  estabelecimentos.parquet (opcional) cnes, nome, tipo, municipio_codigo
  realiza.parquet        (opcional) cnes, procedimento_codigo, ano, volume

Uso:
    docker compose up -d
    python src/grafo.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "data" / "processed"
LOTE = 5000

CONSTRAINTS = [
    "CREATE CONSTRAINT regiao_codigo IF NOT EXISTS FOR (r:RegiaoSaude) REQUIRE r.codigo IS UNIQUE",
    "CREATE CONSTRAINT municipio_codigo IF NOT EXISTS FOR (m:Municipio) REQUIRE m.codigo IS UNIQUE",
    "CREATE CONSTRAINT estab_cnes IF NOT EXISTS FOR (e:Estabelecimento) REQUIRE e.cnes IS UNIQUE",
    "CREATE CONSTRAINT proc_codigo IF NOT EXISTS FOR (p:Procedimento) REQUIRE p.codigo IS UNIQUE",
]


def driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
    )


def _ler(nome):
    fp = PROC / nome
    return pd.read_parquet(fp) if fp.exists() else None


def _executar_lotes(sess, cypher, registros):
    for i in range(0, len(registros), LOTE):
        sess.run(cypher, linhas=registros[i : i + LOTE])


def carregar(sess):
    # limpa e cria restricoes
    sess.run("MATCH (n) DETACH DELETE n")
    for c in CONSTRAINTS:
        sess.run(c)

    # regioes
    reg = _ler("regioes.parquet")
    if reg is not None:
        _executar_lotes(
            sess,
            """
            UNWIND $linhas AS l
            MERGE (r:RegiaoSaude {codigo: l.codigo})
            SET r.nome = l.nome, r.drs = l.drs, r.polo_oncologico = l.polo_oncologico
            """,
            reg.to_dict("records"),
        )
        print(f"  RegiaoSaude: {len(reg)}")

    # municipios + PERTENCE_A
    mun = _ler("municipios.parquet")
    if mun is not None:
        _executar_lotes(
            sess,
            """
            UNWIND $linhas AS l
            MERGE (m:Municipio {codigo: l.codigo})
            SET m.nome = l.nome, m.populacao = l.populacao,
                m.idhm_renda = l.idhm_renda, m.dist_polo_km = l.dist_polo_km
            WITH m, l
            MATCH (r:RegiaoSaude {codigo: l.regiao_codigo})
            MERGE (m)-[:PERTENCE_A]->(r)
            """,
            mun.to_dict("records"),
        )
        print(f"  Municipio: {len(mun)} (+PERTENCE_A)")

    # fluxo (aresta central)
    flu = _ler("fluxo.parquet")
    if flu is not None:
        _executar_lotes(
            sess,
            """
            UNWIND $linhas AS l
            MERGE (o:RegiaoSaude {codigo: l.reg_res_codigo})
              ON CREATE SET o.nome = l.reg_res_nome
            MERGE (d:RegiaoSaude {codigo: l.reg_aten_codigo})
              ON CREATE SET d.nome = l.reg_aten_nome
            CREATE (o)-[:FLUXO {ano: l.ano, complexidade: l.complexidade,
                    linha: l.linha, volume: l.volume, deslocou: l.deslocou}]->(d)
            """,
            flu.to_dict("records"),
        )
        print(f"  FLUXO: {len(flu)} arestas")

    # camada opcional CNES
    proc = _ler("procedimentos.parquet")
    if proc is not None:
        _executar_lotes(
            sess,
            """
            UNWIND $linhas AS l
            MERGE (p:Procedimento {codigo: l.codigo})
            SET p.nome = l.nome, p.grupo = l.grupo, p.linha = l.linha,
                p.complexidade = l.complexidade, p.cid_grupo = l.cid_grupo
            """,
            proc.to_dict("records"),
        )
        print(f"  Procedimento: {len(proc)}")

    est = _ler("estabelecimentos.parquet")
    if est is not None:
        _executar_lotes(
            sess,
            """
            UNWIND $linhas AS l
            MERGE (e:Estabelecimento {cnes: l.cnes})
            SET e.nome = l.nome, e.tipo = l.tipo
            WITH e, l
            MATCH (m:Municipio {codigo: l.municipio_codigo})
            MERGE (e)-[:LOCALIZADO_EM]->(m)
            """,
            est.to_dict("records"),
        )
        print(f"  Estabelecimento: {len(est)} (+LOCALIZADO_EM)")

    rea = _ler("realiza.parquet")
    if rea is not None:
        _executar_lotes(
            sess,
            """
            UNWIND $linhas AS l
            MATCH (e:Estabelecimento {cnes: l.cnes})
            MATCH (p:Procedimento {codigo: l.procedimento_codigo})
            MERGE (e)-[r:REALIZA {ano: l.ano}]->(p)
            SET r.volume = l.volume
            """,
            rea.to_dict("records"),
        )
        print(f"  REALIZA: {len(rea)} arestas")


def main():
    if not (PROC / "fluxo.parquet").exists():
        print(f"Nao encontrei {PROC/'fluxo.parquet'}. Rode antes: python src/extracao.py")
        sys.exit(1)
    d = driver()
    try:
        d.verify_connectivity()
    except Exception as e:  # noqa: BLE001
        print(f"Neo4j indisponivel: {e}\nSuba com 'docker compose up -d'.")
        sys.exit(1)
    print("Carregando grafo...")
    with d.session() as sess:
        carregar(sess)
        n = sess.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        r = sess.run("MATCH ()-[x]->() RETURN count(x) AS r").single()["r"]
    d.close()
    print(f"Pronto: {n} nos, {r} relacoes.")


if __name__ == "__main__":
    main()
