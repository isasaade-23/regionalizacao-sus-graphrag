"""
esquema.py — descricao do grafo em texto (para o prompt do LLM) e exemplos
semente de pergunta->Cypher. Fonte unica usada pelo Text2Cypher e pela geracao
do dataset sintetico. O esquema completo esta em schema/grafo.md.
"""

# Resumo do esquema injetado no prompt do LLM (compacto de proposito).
ESQUEMA_TEXTO = """\
Grafo Neo4j do fluxo assistencial do SUS (recorte oncologico, Sao Paulo).

Nos:
  (:RegiaoSaude {codigo, nome, drs, polo_oncologico})
  (:Municipio {codigo, nome, populacao, idhm_renda, dist_polo_km})
  (:Estabelecimento {cnes, nome, tipo})
  (:Procedimento {codigo, nome, grupo, linha, complexidade, cid_grupo})

Relacoes:
  (:Municipio)-[:PERTENCE_A]->(:RegiaoSaude)
  (:RegiaoSaude)-[:FLUXO {ano, complexidade, linha, volume, deslocou}]->(:RegiaoSaude)
      // origem = regiao de residencia; destino = regiao de atendimento
      // deslocou=true quando origem<>destino; laco (origem=destino) = retencao
  (:Estabelecimento)-[:LOCALIZADO_EM]->(:Municipio)
  (:Estabelecimento)-[:REALIZA {ano, volume}]->(:Procedimento)

Convencoes:
  - complexidade em {'basica','media','alta'}
  - linha em {'radioterapia','quimioterapia','diagnostico'}
  - evasao de uma regiao = soma de FLUXO com deslocou=true saindo dela
  - invasao de uma regiao = soma de FLUXO com deslocou=true chegando nela
  - IRR (retencao) = volume retido (deslocou=false) / volume total demandado pelos residentes
"""

# Exemplos semente (few-shot). Alimentam o prompt do Text2Cypher e servem de
# ponto de partida para a destilacao do dataset de fine-tuning.
EXEMPLOS = [
    {
        "pergunta": "Quais regioes de saude mais enviam pacientes de cancer para fora?",
        "cypher": (
            "MATCH (o:RegiaoSaude)-[f:FLUXO]->(:RegiaoSaude) "
            "WHERE f.deslocou = true "
            "RETURN o.nome AS regiao, sum(f.volume) AS evasao "
            "ORDER BY evasao DESC LIMIT 10"
        ),
    },
    {
        "pergunta": "Qual polo mais recebe pacientes de fora para radioterapia?",
        "cypher": (
            "MATCH (:RegiaoSaude)-[f:FLUXO]->(d:RegiaoSaude) "
            "WHERE f.deslocou = true AND f.linha = 'radioterapia' "
            "RETURN d.nome AS polo, sum(f.volume) AS invasao "
            "ORDER BY invasao DESC LIMIT 5"
        ),
    },
    {
        "pergunta": "Qual o indice de retencao regional da regiao de Aracatuba na alta complexidade?",
        "cypher": (
            "MATCH (o:RegiaoSaude {nome: 'Aracatuba'})-[f:FLUXO]->(:RegiaoSaude) "
            "WHERE f.complexidade = 'alta' "
            "WITH sum(CASE WHEN f.deslocou = false THEN f.volume ELSE 0 END) AS retido, "
            "sum(f.volume) AS total "
            "RETURN retido * 1.0 / total AS irr"
        ),
    },
    {
        "pergunta": "Quais municipios da regiao de Registro ficam mais longe do polo?",
        "cypher": (
            "MATCH (m:Municipio)-[:PERTENCE_A]->(:RegiaoSaude {nome: 'Registro'}) "
            "RETURN m.nome AS municipio, m.dist_polo_km AS distancia "
            "ORDER BY distancia DESC LIMIT 10"
        ),
    },
    {
        "pergunta": "Quantos atendimentos de quimioterapia cada regiao reteve em 2024?",
        "cypher": (
            "MATCH (o:RegiaoSaude)-[f:FLUXO]->(o) "
            "WHERE f.linha = 'quimioterapia' AND f.ano = 2024 AND f.deslocou = false "
            "RETURN o.nome AS regiao, sum(f.volume) AS retido "
            "ORDER BY retido DESC"
        ),
    },
    {
        "pergunta": "Quais estabelecimentos realizam radioterapia e em que municipio ficam?",
        "cypher": (
            "MATCH (e:Estabelecimento)-[:REALIZA]->(p:Procedimento {linha: 'radioterapia'}), "
            "(e)-[:LOCALIZADO_EM]->(m:Municipio) "
            "RETURN e.nome AS estabelecimento, m.nome AS municipio"
        ),
    },
]

# Prompt base do Text2Cypher (portugues). O {esquema} e as {pergunta} sao
# preenchidos em tempo de execucao.
PROMPT_TEXT2CYPHER = """\
Voce traduz perguntas em portugues para consultas Cypher sobre um grafo Neo4j.
Use SOMENTE os rotulos, relacoes e propriedades do esquema abaixo. Nao invente
campos. Responda com UMA consulta Cypher, sem explicacao e sem cercas de codigo.

ESQUEMA:
{esquema}

EXEMPLOS:
{exemplos}

PERGUNTA: {pergunta}
CYPHER:"""


def montar_prompt(pergunta, exemplos=None):
    """Monta o prompt few-shot do Text2Cypher."""
    exs = exemplos if exemplos is not None else EXEMPLOS
    bloco = "\n".join(f"P: {e['pergunta']}\nC: {e['cypher']}" for e in exs)
    return PROMPT_TEXT2CYPHER.format(
        esquema=ESQUEMA_TEXTO, exemplos=bloco, pergunta=pergunta
    )
