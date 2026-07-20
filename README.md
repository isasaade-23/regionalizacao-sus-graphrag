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

## Arquitetura

```mermaid
flowchart LR
    P["Pergunta em<br/>portugues natural"] --> G

    subgraph Traducao["Text2Cypher (LLM)"]
        G{"Provedor"}
        G -->|gemini / groq| A["API"]
        G -->|local| Q["Modelo aberto<br/>+ adapter QLoRA"]
    end

    A --> C["Consulta Cypher"]
    Q --> C
    C --> N[("Neo4j<br/>grafo do fluxo<br/>assistencial")]
    N --> R["Resposta<br/>tabular"]

    subgraph Offline["Preparacao (offline)"]
        D["SIA/DataSUS"] --> E["extracao.py"] --> Grafo["grafo.py"] --> N
        Sem["perguntas semente<br/>+ schema"] --> GD["gerar_dataset.py<br/>(destilacao via Gemini)"] --> DS["1.500 pares<br/>pergunta -> Cypher"] --> Q
    end

    N -. "EXPLAIN + execucao" .-> AV["avaliar.py<br/>validade + acerto"]
```

O caminho de consulta (topo) traduz a pergunta em Cypher por um dos tres
provedores, executa no Neo4j e devolve a resposta. A preparacao (offline) carrega
o grafo a partir do SIA e destila o dataset que treina o modelo aberto com QLoRA.
`avaliar.py` fecha o ciclo de MLOps comparando os provedores no mesmo conjunto.

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

## Avaliacao

A comparacao alvo e entre dois caminhos de traducao pergunta->Cypher: a API
(professor) e um modelo aberto pequeno destilado com QLoRA (aluno). O harness
`src/avaliar.py` mede a metrica que de fato importa — **validade** (o Cypher passa
no `EXPLAIN` do Neo4j) e **acerto de execucao** (rodado, retorna o mesmo resultado
que o Cypher de referencia) —, porque duas consultas escritas de formas diferentes
podem estar as duas corretas. Essa medida exige o grafo carregado e e o passo
reprodutivel seguinte: `docker compose up` + `src/grafo.py` + `src/avaliar.py`.

**Fine-tuning (medido).** O modelo aberto **Qwen2.5-1.5B**, ajustado com LoRA sobre
1.500 pares pergunta->Cypher destilados do Gemini, foi avaliado em um conjunto de
validacao de 150 perguntas (split fixo, `seed=42`) por correspondencia ao Cypher de
referencia. O treino rodou na GPU gratuita do Kaggle e exigiu varias iteracoes para
estabilizar o ambiente (bitsandbytes/CUDA na GPU sorteada), caindo para LoRA em
fp16 com o bloco QLoRA 4-bit documentado como opcao.

| Modelo | Acerto exato | Acerto estrutural | n |
|---|---|---|---|
| Qwen2.5-1.5B + LoRA (destilado, 1.500 pares) | 6,0% | 8,7% | 150 |

"Acerto estrutural" ignora nomes de variavel (compara rotulos, relacoes,
propriedades e palavras-chave). Os numeros sao modestos e esperados para um modelo
de 1,5B destilado com poucos milhares de pares: ele aprende o esquema do grafo mas
ainda trunca antes de fechar `WHERE`/`RETURN`. Mais dados de destilacao e trocar o
alvo de string por execucao sao as alavancas seguintes.

**API (Gemini) — evidencia qualitativa.** Nas perguntas curadas de `eval/`, o Gemini
gera Cypher valido e correto, proximo do gabarito (~1k tokens por consulta). A taxa
formal por execucao entra na tabela assim que o grafo e carregado. A correspondencia
por string subestima a API, que escreve consultas corretas porem estilisticamente
diferentes da referencia — exatamente por isso a metrica de execucao e a correta.

> **Dados:** o exemplo roda sobre uma **amostra sintetica representativa** (20
> regioes, 127 municipios, ~2 mil arestas de fluxo, ~30% de deslocamento, retencao
> caindo com a complexidade). O caminho de extracao do SIA/DataSUS real existe em
> `src/extracao.py`; a amostra mantem o repositorio leve e reprodutivel.

## Fontes de dados

Todas publicas: SIA/SUS via FTP do DataSUS; malha oficial municipio->regiao de
saude (DATASUS/e-Gestor); IBGE (populacao) e IPEA (renda) para atributos do
municipio. Dados brutos nao sao versionados (ver `.gitignore`).

## Licenca

MIT. Autoria individual de Isabela Venancio da Silva.
