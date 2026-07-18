# regionalizacao-sus-graphrag

**Grafo de conhecimento e GraphRAG em portugues sobre o fluxo assistencial do SUS,
com a jornada oncologica (cancer de mama) como aplicacao.**

O repositorio modela o deslocamento de pacientes entre municipios e regioes de
saude a partir de dados publicos do DataSUS, carrega esse fluxo num grafo Neo4j e
permite perguntar em portugues natural, com a pergunta traduzida para Cypher por
LLM (GraphRAG). Um modelo pequeno e aberto e ajustado com QLoRA para essa traducao
e comparado com a API.

## O que este repositorio comprova

| Requisito | Onde se comprova |
|---|---|
| **Fine-tuning LoRA/QLoRA** | `notebooks/02_qlora_kaggle.ipynb` — destila pares pergunta->Cypher do Gemini e treina um modelo aberto com QLoRA |
| **LLM + Grafos de Conhecimento** | `src/grafo.py` + `docker-compose.yml` (Neo4j) — carga do fluxo assistencial como grafo |
| **NLP em portugues** | `src/text2cypher.py` — pergunta em portugues -> Cypher -> resposta |
| **MLOps e avaliacao de modelos** | `src/avaliar.py` + `docker-compose.yml` — validade e acerto do Cypher; ambiente reprodutivel |

## Dominio

O SUS se organiza em regioes de saude. Quando residencia e local de atendimento
pertencem a regioes distintas, ha **deslocamento inter-regional**. O cancer serve
como condicao traçadora: depende de alta complexidade, concentra-se em polos
estaduais e tem bom registro na Autorizacao de Procedimentos de Alta Complexidade
(APAC). O grafo torna esse fluxo consultavel: quais regioes evadem, quais polos
concentram a atracao, por complexidade e por linha de cuidado.

Recorte do exemplo: estado de Sao Paulo, SIA/SUS 2019 a 2024, dados publicos.

## Estrutura

```
.
├── schema/grafo.md          # esquema do grafo (nos, relacoes, propriedades)
├── src/
│   ├── extracao.py          # DataSUS (SIA) -> tabela de fluxo
│   ├── grafo.py             # tabela de fluxo -> Neo4j
│   ├── text2cypher.py       # pergunta PT -> Cypher -> resposta (API e modelo local)
│   ├── gerar_dataset.py     # dataset sintetico pergunta->Cypher via Gemini (destilacao)
│   └── avaliar.py           # validade e acerto do Cypher gerado
├── notebooks/
│   ├── 01_exploracao.ipynb  # fluxo assistencial: exploracao dos dados
│   └── 02_qlora_kaggle.ipynb# fine-tuning QLoRA (GPU gratuita do Kaggle)
├── eval/perguntas.yaml      # conjunto de avaliacao: pergunta + Cypher de referencia
├── docker-compose.yml       # Neo4j
└── data/                    # raw/ e processed/ (fora do versionamento)
```

## Como rodar

```bash
cp .env.example .env        # preencha as chaves
docker compose up -d        # sobe o Neo4j
pip install -r requirements.txt

python src/extracao.py      # baixa e prepara o fluxo (SIA/SP)
python src/grafo.py         # carrega o grafo no Neo4j
python src/text2cypher.py "Quais regioes de saude mais enviam pacientes de cancer para fora?"
python src/avaliar.py       # roda o conjunto de avaliacao
```

O fine-tuning roda a parte, no Kaggle: abra `notebooks/02_qlora_kaggle.ipynb`,
gere o dataset com `src/gerar_dataset.py`, treine e traga o adapter de volta.

## Comparacao API vs. modelo fine-tunado

A tabela final (validade de Cypher e acerto de execucao no conjunto `eval/`) e
preenchida por `src/avaliar.py` apos rodar os dois caminhos.

| Modelo | Cypher valido (%) | Resposta correta (%) | Custo por consulta |
|---|---|---|---|
| Gemini (API) | _a preencher_ | _a preencher_ | _a preencher_ |
| Modelo aberto + QLoRA | _a preencher_ | _a preencher_ | _a preencher_ |

## Fontes de dados

Todas publicas: SIA/SUS via FTP do DataSUS; malha oficial municipio->regiao de
saude (DATASUS/e-Gestor); IBGE (populacao) e IPEA (renda) para atributos do
municipio. Dados brutos nao sao versionados (ver `.gitignore`).

## Licenca

MIT. Autoria individual de Isabela Venancio da Silva.
