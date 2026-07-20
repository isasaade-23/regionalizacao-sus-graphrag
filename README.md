# regionalizacao-sus-graphrag

**Grafo de conhecimento e GraphRAG em português sobre o fluxo assistencial do SUS, com a jornada oncológica (câncer de mama) como aplicação.**

O repositório modela o deslocamento de pacientes entre municípios e regiões de saúde a partir de dados públicos do DataSUS, carrega esse fluxo num grafo Neo4j e permite perguntar em português natural, com a pergunta traduzida para Cypher por LLM (GraphRAG). Um modelo pequeno e aberto é ajustado com QLoRA para essa tradução e comparado com a API.

## O que este repositório comprova

| Requisito | Onde se comprova |
| --- | --- |
| **Fine-tuning LoRA/QLoRA** | `notebooks/02_qlora_kaggle.ipynb` — destila pares pergunta→Cypher do Gemini e treina um modelo aberto com QLoRA |
| **LLM + Grafos de Conhecimento** | `src/grafo.py` + `docker-compose.yml` (Neo4j) — carga do fluxo assistencial como grafo |
| **NLP em português** | `src/text2cypher.py` — pergunta em português → Cypher → resposta |
| **MLOps e avaliação de modelos** | `src/avaliar.py` + `docker-compose.yml` — validade e acerto do Cypher; ambiente reprodutível |

## Arquitetura

```mermaid
flowchart LR
    P["Pergunta em<br/>português natural"] --> G

    subgraph Traducao["Text2Cypher (LLM)"]
        G{"Provedor"}
        G -->|gemini / groq| A["API"]
        G -->|local| Q["Modelo aberto<br/>+ adapter QLoRA"]
    end

    A --> C["Consulta Cypher"]
    Q --> C
    C --> N[("Neo4j<br/>grafo do fluxo<br/>assistencial")]
    N --> R["Resposta<br/>tabular"]

    subgraph Offline["Preparação (offline)"]
        D["SIA/DataSUS"] --> E["extracao.py"] --> Grafo["grafo.py"] --> N
        Sem["perguntas semente<br/>+ schema"] --> GD["gerar_dataset.py<br/>(destilação via Gemini)"] --> DS["1.500 pares<br/>pergunta → Cypher"] --> Q
    end

    N -. "EXPLAIN + execução" .-> AV["avaliar.py<br/>validade + acerto"]
```

O caminho de consulta (topo) traduz a pergunta em Cypher por um dos três provedores, executa no Neo4j e devolve a resposta. A preparação (offline) carrega o grafo a partir do SIA e destila o dataset que treina o modelo aberto com QLoRA. `avaliar.py` fecha o ciclo de MLOps comparando os provedores no mesmo conjunto.

## Domínio

O SUS se organiza em regiões de saúde. Quando residência e local de atendimento pertencem a regiões distintas, há **deslocamento inter-regional**. O câncer serve como condição traçadora: depende de alta complexidade, concentra-se em polos estaduais e tem bom registro na Autorização de Procedimentos de Alta Complexidade (APAC). O grafo torna esse fluxo consultável: quais regiões evadem, quais polos concentram a atração, por complexidade e por linha de cuidado.

Recorte do exemplo: estado de São Paulo, SIA/SUS 2019 a 2024, dados públicos.

## Estrutura

```
.
├── schema/grafo.md          # esquema do grafo (nós, relações, propriedades)
├── src/
│   ├── extracao.py          # DataSUS (SIA) -> tabela de fluxo
│   ├── grafo.py             # tabela de fluxo -> Neo4j
│   ├── text2cypher.py       # pergunta PT -> Cypher -> resposta (API e modelo local)
│   ├── gerar_dataset.py     # dataset sintético pergunta->Cypher via Gemini (destilação)
│   └── avaliar.py           # validade e acerto do Cypher gerado
├── notebooks/
│   ├── 01_exploracao.ipynb  # fluxo assistencial: exploração dos dados
│   └── 02_qlora_kaggle.ipynb# fine-tuning QLoRA (GPU gratuita do Kaggle)
├── eval/perguntas.yaml      # conjunto de avaliação: pergunta + Cypher de referência
├── docker-compose.yml       # Neo4j
└── data/                    # raw/ e processed/ (fora do versionamento)
```

## Como rodar

```
cp .env.example .env        # preencha as chaves
docker compose up -d        # sobe o Neo4j
pip install -r requirements.txt

python src/extracao.py      # baixa e prepara o fluxo (SIA/SP)
python src/grafo.py         # carrega o grafo no Neo4j
python src/text2cypher.py "Quais regiões de saúde mais enviam pacientes de câncer para fora?"
python src/avaliar.py       # roda o conjunto de avaliação
```

O fine-tuning roda à parte, no Kaggle: abra `notebooks/02_qlora_kaggle.ipynb`, gere o dataset com `src/gerar_dataset.py`, treine e traga o adapter de volta.

## Avaliação

A comparação alvo é entre dois caminhos de tradução pergunta→Cypher: a API (professor) e um modelo aberto pequeno destilado com QLoRA (aluno). O harness `src/avaliar.py` mede a métrica que de fato importa — **validade** (o Cypher passa no `EXPLAIN` do Neo4j) e **acerto de execução** (rodado, retorna o mesmo resultado que o Cypher de referência) —, porque duas consultas escritas de formas diferentes podem estar as duas corretas. Essa medida exige o grafo carregado e é o passo reprodutível seguinte: `docker compose up` + `src/grafo.py` + `src/avaliar.py`.

**Fine-tuning (medido).** O modelo aberto **Qwen2.5-1.5B**, ajustado com LoRA sobre 1.500 pares pergunta→Cypher destilados do Gemini, foi avaliado em um conjunto de validação de 150 perguntas (split fixo, `seed=42`) por correspondência ao Cypher de referência. O treino rodou na GPU gratuita do Kaggle e exigiu várias iterações para estabilizar o ambiente (bitsandbytes/CUDA na GPU sorteada), caindo para LoRA em fp16 com o bloco QLoRA 4-bit documentado como opção.

| Modelo | Acerto exato | Acerto estrutural | n |
| --- | --- | --- | --- |
| Qwen2.5-1.5B + LoRA (destilado, 1.500 pares) | 6,0% | 8,7% | 150 |

"Acerto estrutural" ignora nomes de variável (compara rótulos, relações, propriedades e palavras-chave). Os números são modestos e esperados para um modelo de 1,5B destilado com poucos milhares de pares: ele aprende o esquema do grafo mas ainda trunca antes de fechar `WHERE`/`RETURN`. Mais dados de destilação e trocar o alvo de string por execução são as alavancas seguintes.

**API (Gemini) — evidência qualitativa.** Nas perguntas curadas de `eval/`, o Gemini gera Cypher válido e correto, próximo do gabarito (~1k tokens por consulta). A taxa formal por execução entra na tabela assim que o grafo é carregado. A correspondência por string subestima a API, que escreve consultas corretas porém estilisticamente diferentes da referência — exatamente por isso a métrica de execução é a correta.

> **Dados:** o exemplo roda sobre uma **amostra sintética representativa** (20 regiões, 127 municípios, ~2 mil arestas de fluxo, ~30% de deslocamento, retenção caindo com a complexidade). O caminho de extração do SIA/DataSUS real existe em `src/extracao.py`; a amostra mantém o repositório leve e reprodutível.

## Fontes de dados

Todas públicas: SIA/SUS via FTP do DataSUS; malha oficial município→região de saúde (DATASUS/e-Gestor); IBGE (população) e IPEA (renda) para atributos do município. Dados brutos não são versionados (ver `.gitignore`).

## Licença

MIT. Autoria individual de Isabela Venancio da Silva.
