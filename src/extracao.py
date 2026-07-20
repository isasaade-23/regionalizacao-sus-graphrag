"""
extracao.py — DataSUS (SIA/SUS) -> tabela de fluxo assistencial (recorte onco).

Dois modos:

  --amostra   Gera uma amostra sintetica, porem realista, das tabelas tidy, para
              rodar o grafo e o Text2Cypher de ponta a ponta sem baixar gigabytes.
              Recomendado para experimentar o repositorio.

  --ano AAAA  Extracao real do SIA via pysus (FTP DATASUS). Baixa a producao
              ambulatorial de SP, filtra o recorte oncologico (grupo SIGTAP 0304
              e CID de neoplasia maligna), restringe aos registros com residencia
              informada, mapeia municipio->regiao pela malha oficial e agrega o
              fluxo por (regiao_residencia, regiao_atendimento, ano, complexidade,
              linha). Reaproveita a mesma operacionalizacao do artigo metodologico.

Saidas em data/processed/: regioes.parquet, municipios.parquet, fluxo.parquet e,
no modo amostra, a camada CNES (procedimentos/estabelecimentos/realiza).

Uso:
    python src/extracao.py --amostra
    python src/extracao.py --ano 2024
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "data" / "processed"
INTERIM = RAIZ / "data" / "interim"

# Recorte tracador (ver schema/grafo.md e o artigo metodologico)
GRUPO_ONCO = "0304"       # radioterapia e quimioterapia (SIGTAP)
LINHAS = {"radioterapia": "0304", "quimioterapia": "0304", "diagnostico": "0201"}
COMPLEXIDADES = ["basica", "media", "alta"]


# ─────────────────────────────────────────────
# MODO AMOSTRA (sintetico, deterministico)
# ─────────────────────────────────────────────
def gerar_amostra(seed=42):
    rng = np.random.default_rng(seed)

    # 20 regioes de saude, 5 delas polos oncologicos
    nomes_reg = [
        "Sao Paulo", "Campinas", "Ribeirao Preto", "Sao Jose do Rio Preto", "Botucatu",
        "Aracatuba", "Presidente Prudente", "Marilia", "Bauru", "Sorocaba",
        "Registro", "Taubate", "Sao Jose dos Campos", "Franca", "Aracatuba Oeste",
        "Piracicaba", "Braganca", "Barretos", "Aracatuba Leste", "Litoral Norte",
    ]
    polos = {"Sao Paulo", "Campinas", "Ribeirao Preto", "Barretos", "Botucatu"}
    regioes = pd.DataFrame({
        "codigo": [f"350{str(i+1).zfill(2)}" for i in range(len(nomes_reg))],
        "nome": nomes_reg,
        "drs": [f"DRS-{(i % 17) + 1:02d}" for i in range(len(nomes_reg))],
        "polo_oncologico": [n in polos for n in nomes_reg],
    })

    # ~6 municipios por regiao
    mun_rows = []
    cod = 350000
    for _, r in regioes.iterrows():
        n_mun = int(rng.integers(4, 9))
        for j in range(n_mun):
            cod += 1
            eh_polo_mun = (j == 0 and r["polo_oncologico"])
            mun_rows.append({
                "codigo": str(cod),
                "nome": f"{r['nome']} {'(polo)' if eh_polo_mun else j+1}",
                "regiao_codigo": r["codigo"],
                "populacao": int(rng.integers(15_000, 900_000) if not eh_polo_mun else rng.integers(400_000, 12_000_000)),
                "idhm_renda": round(float(rng.uniform(0.62, 0.85)), 3),
                "dist_polo_km": 0.0 if eh_polo_mun else round(float(rng.uniform(8, 180)), 1),
            })
    municipios = pd.DataFrame(mun_rows)

    # fluxo: para cada regiao de residencia x ano x complexidade x linha, parte fica,
    # parte vai para um polo. Retencao alta na basica, baixa na alta (como no artigo).
    ret_por_cplx = {"basica": 0.99, "media": 0.71, "alta": 0.66}
    cod_polos = regioes.loc[regioes["polo_oncologico"], "codigo"].tolist()
    nome_por_cod = dict(zip(regioes["codigo"], regioes["nome"]))
    linhas = []
    for _, r in regioes.iterrows():
        for ano in range(2019, 2025):
            for cplx in COMPLEXIDADES:
                for linha in ["radioterapia", "quimioterapia", "diagnostico"]:
                    demanda = int(rng.integers(80, 1200))
                    ret = ret_por_cplx[cplx] * (1.0 if r["polo_oncologico"] else 0.85)
                    v_ret = int(demanda * ret)
                    v_fora = demanda - v_ret
                    if v_ret > 0:
                        linhas.append(dict(
                            reg_res_codigo=r["codigo"], reg_res_nome=r["nome"],
                            reg_aten_codigo=r["codigo"], reg_aten_nome=r["nome"],
                            ano=ano, complexidade=cplx, linha=linha, volume=v_ret, deslocou=False))
                    if v_fora > 0:
                        destino = cod_polos[int(rng.integers(0, len(cod_polos)))]
                        if destino == r["codigo"]:
                            destino = cod_polos[(cod_polos.index(destino) + 1) % len(cod_polos)]
                        linhas.append(dict(
                            reg_res_codigo=r["codigo"], reg_res_nome=r["nome"],
                            reg_aten_codigo=destino, reg_aten_nome=nome_por_cod[destino],
                            ano=ano, complexidade=cplx, linha=linha, volume=v_fora, deslocou=True))
    fluxo = pd.DataFrame(linhas)

    # camada CNES (oferta): procedimentos tracadores, centros de oncologia nos
    # municipios-polo das regioes polo, e a producao (REALIZA) por ano.
    procedimentos = pd.DataFrame([
        {"codigo": "0304010010", "nome": "Radioterapia externa", "grupo": "0304",
         "linha": "radioterapia", "complexidade": "alta", "cid_grupo": "C50"},
        {"codigo": "0304020010", "nome": "Quimioterapia de neoplasia de mama", "grupo": "0304",
         "linha": "quimioterapia", "complexidade": "alta", "cid_grupo": "C50"},
        {"codigo": "0201010010", "nome": "Biopsia de mama", "grupo": "0201",
         "linha": "diagnostico", "complexidade": "media", "cid_grupo": "C50"},
        {"codigo": "0204030030", "nome": "Mamografia bilateral", "grupo": "0204",
         "linha": "diagnostico", "complexidade": "media", "cid_grupo": "C50"},
    ])

    polo_mun = municipios[municipios["nome"].str.endswith("(polo)")].reset_index(drop=True)
    est_rows, rea_rows = [], []
    for i, m in polo_mun.iterrows():
        cnes = str(2077000 + i)
        est_rows.append({
            "cnes": cnes,
            "nome": f"Centro de Oncologia de {m['nome'].replace(' (polo)', '')}",
            "tipo": "unidade de alta complexidade em oncologia",
            "municipio_codigo": m["codigo"],
        })
        for _, p in procedimentos.iterrows():
            for ano in range(2019, 2025):
                rea_rows.append({
                    "cnes": cnes, "procedimento_codigo": p["codigo"],
                    "ano": ano, "volume": int(rng.integers(200, 5000)),
                })
    estabelecimentos = pd.DataFrame(est_rows)
    realiza = pd.DataFrame(rea_rows)

    return regioes, municipios, fluxo, procedimentos, estabelecimentos, realiza


# ─────────────────────────────────────────────
# MODO REAL (pysus / FTP DATASUS)
# ─────────────────────────────────────────────
def _mapa_municipio_regiao():
    """Malha oficial municipio(6 dig) -> regiao de saude(5 dig).
    Espera data/interim/rl_municip_regsaud.csv (CO_MUNICIP;CO_REGSAUD), como no
    projeto de origem. Baixe do e-Gestor/DATASUS."""
    fp = INTERIM / "rl_municip_regsaud.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"Malha municipio->regiao ausente: {fp}. "
            "Baixe a tabela oficial (CO_MUNICIP;CO_REGSAUD) do e-Gestor/DATASUS."
        )
    m = pd.read_csv(fp, sep=";", dtype=str)
    m.columns = [c.strip().upper() for c in m.columns]
    return dict(zip(m["CO_MUNICIP"].str[:6], m["CO_REGSAUD"]))


def extrair_ano(ano):
    """Extracao real do SIA de SP para um ano, recorte oncologico -> fluxo."""
    try:
        from pysus.online_data.SIA import download
    except ImportError:
        raise ImportError("pysus nao instalado. Rode: pip install pysus")

    mapa = _mapa_municipio_regiao()
    print(f"[{ano}] baixando SIA-PA de SP (12 competencias)...")
    # PA = producao ambulatorial; APAC (AR/AQ) tras residencia para alta complexidade.
    partes = []
    for grupo in ["PA", "AR", "AQ"]:
        try:
            df = download("SP", ano, list(range(1, 13)), group=grupo)
            partes.append(_preparar_bruto(df, mapa))
        except Exception as e:  # noqa: BLE001
            print(f"  grupo {grupo}: {str(e)[:80]}")
    if not partes:
        raise RuntimeError("Nenhum dado baixado.")
    dados = pd.concat(partes, ignore_index=True)
    dados["ano"] = ano
    fluxo = (
        dados.groupby(
            ["reg_res_codigo", "reg_aten_codigo", "ano", "complexidade", "linha"], as_index=False
        )["volume"].sum()
    )
    # anexa nomes e flag de deslocamento
    fluxo["deslocou"] = fluxo["reg_res_codigo"] != fluxo["reg_aten_codigo"]
    return fluxo


def _preparar_bruto(df, mapa):
    """Filtra recorte onco, mapeia regiao, deriva linha/complexidade. Espera colunas
    do SIA: PA_PROC_ID, PA_CIDPRI, PA_MUNPCN (residencia), PA_UFMUN (estab),
    PA_DOCORIG, PA_MN_IND, PA_QTDAPR."""
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    proc = df.get("PA_PROC_ID", pd.Series(dtype=str)).astype(str)
    cid = df.get("PA_CIDPRI", pd.Series(dtype=str)).astype(str)
    onco = proc.str[:4].eq(GRUPO_ONCO) | cid.str.startswith("C")
    # residencia informada
    resid_ok = df.get("PA_MN_IND", pd.Series(dtype=str)).astype(str).eq("M")
    df = df[onco & resid_ok].copy()

    mun_res = df["PA_MUNPCN"].astype(str).str[:6]
    mun_ate = df["PA_UFMUN"].astype(str).str[:6]
    df["reg_res_codigo"] = mun_res.map(mapa)
    df["reg_aten_codigo"] = mun_ate.map(mapa)
    df = df.dropna(subset=["reg_res_codigo", "reg_aten_codigo"])

    df["linha"] = np.where(proc.loc[df.index].str[:4].eq("0304"),
                           np.where(proc.loc[df.index].str.contains("RADIO", case=False), "radioterapia", "quimioterapia"),
                           "diagnostico")
    nivel = proc.loc[df.index].str[:2]
    df["complexidade"] = np.select(
        [nivel.eq("03"), nivel.eq("02")], ["alta", "media"], default="basica"
    )
    df["volume"] = 1  # contagem por evento (registro), nao por quantidade aprovada
    return df[["reg_res_codigo", "reg_aten_codigo", "linha", "complexidade", "volume"]]


# ─────────────────────────────────────────────
def salvar(regioes, municipios, fluxo, procedimentos=None, estabelecimentos=None, realiza=None):
    PROC.mkdir(parents=True, exist_ok=True)
    if regioes is not None:
        regioes.to_parquet(PROC / "regioes.parquet", index=False)
    if municipios is not None:
        municipios.to_parquet(PROC / "municipios.parquet", index=False)
    fluxo.to_parquet(PROC / "fluxo.parquet", index=False)
    # camada CNES opcional
    for df, nome in [(procedimentos, "procedimentos"), (estabelecimentos, "estabelecimentos"),
                     (realiza, "realiza")]:
        if df is not None:
            df.to_parquet(PROC / f"{nome}.parquet", index=False)
    print(f"Salvo em {PROC}: fluxo={len(fluxo)} arestas"
          + (f", regioes={len(regioes)}" if regioes is not None else "")
          + (f", municipios={len(municipios)}" if municipios is not None else "")
          + (f", estabelecimentos={len(estabelecimentos)}" if estabelecimentos is not None else "")
          + (f", realiza={len(realiza)}" if realiza is not None else ""))


def main():
    ap = argparse.ArgumentParser(description="Extracao SIA/SUS -> fluxo assistencial")
    ap.add_argument("--amostra", action="store_true", help="gera amostra sintetica realista")
    ap.add_argument("--ano", type=int, help="extracao real do SIA para o ano (via pysus)")
    args = ap.parse_args()

    if args.amostra or not args.ano:
        if not args.amostra:
            print("Nenhum --ano informado; gerando --amostra. Use --ano AAAA para dados reais.")
        regioes, municipios, fluxo, procedimentos, estabelecimentos, realiza = gerar_amostra()
        salvar(regioes, municipios, fluxo, procedimentos, estabelecimentos, realiza)
    else:
        fluxo = extrair_ano(args.ano)
        # no modo real, regioes/municipios vem de features_*.py do projeto de origem;
        # aqui salvamos ao menos o fluxo. Complete regioes/municipios conforme suas fontes.
        salvar(None, None, fluxo)


if __name__ == "__main__":
    main()
