# Esquema do grafo — fluxo assistencial do SUS

O grafo representa o **fluxo de pacientes entre regiões de saúde** para
atendimento, com o câncer como condição traçadora. A pergunta central que o
grafo responde: *para onde vão os residentes de cada região quando buscam
atendimento, e quais polos concentram essa atração?*

A unidade territorial de análise é a **região de saúde** (não o município):
deslocamento dentro da própria região é pactuado e esperado; o sinal de
interesse é o residente sair da sua região de saúde para ser atendido em outra.

## Nós

### `RegiaoSaude`
Unidade analítica principal. São Paulo tem 62 regiões de saúde.

| Propriedade | Tipo | Descrição |
|---|---|---|
| `codigo` | string | Código da região (CO_REGSAUD, 5 dígitos). **Chave** |
| `nome` | string | Nome da região de saúde |
| `drs` | string | Departamento Regional de Saúde a que pertence |
| `polo_oncologico` | boolean | Se concentra atendimento oncológico de outras regiões |

### `Municipio`
Unidade de residência do paciente. São Paulo tem 645 municípios.

| Propriedade | Tipo | Descrição |
|---|---|---|
| `codigo` | string | Código IBGE de 6 dígitos (CO_MUNICIP). **Chave** |
| `nome` | string | Nome do município |
| `uf` | string | Unidade federativa (35 = SP) |
| `populacao` | int | População (IBGE/Censo 2022) |
| `idhm_renda` | float | IDHM-Renda (IPEA) |
| `dist_polo_km` | float | Distância ao polo da própria região de saúde |

### `Estabelecimento`
Onde o procedimento é realizado (CNES). Camada opcional, usada para localizar a
oferta que sustenta um polo.

| Propriedade | Tipo | Descrição |
|---|---|---|
| `cnes` | string | Código CNES. **Chave** |
| `nome` | string | Nome fantasia do estabelecimento |
| `tipo` | string | Tipo (hospital, unidade de alta complexidade em oncologia, etc.) |

### `Procedimento`
Procedimento oncológico do SIA/SUS (recorte traçador).

| Propriedade | Tipo | Descrição |
|---|---|---|
| `codigo` | string | Código SIGTAP. **Chave** |
| `nome` | string | Descrição do procedimento |
| `grupo` | string | Grupo SIGTAP (0304 = radioterapia e quimioterapia) |
| `linha` | string | `radioterapia`, `quimioterapia` ou `diagnostico` |
| `complexidade` | string | `basica`, `media` ou `alta` |
| `cid_grupo` | string | Grupo CID de neoplasia maligna associado (ex.: C50 = mama) |

## Relações

### `(Municipio)-[:PERTENCE_A]->(RegiaoSaude)`
Malha oficial município → região de saúde (DATASUS/e-Gestor). Sem propriedades.

### `(RegiaoSaude)-[:FLUXO]->(RegiaoSaude)`
Aresta central. Da região **de residência** para a região **de atendimento**,
agregada por ano, complexidade e linha de cuidado.

| Propriedade | Tipo | Descrição |
|---|---|---|
| `ano` | int | Ano de competência (2019–2024) |
| `complexidade` | string | `basica`, `media`, `alta` |
| `linha` | string | `radioterapia`, `quimioterapia`, `diagnostico` |
| `volume` | int | Número de atendimentos (contagem por evento) |
| `deslocou` | boolean | `true` quando origem ≠ destino (deslocamento inter-regional) |

Quando origem = destino, a aresta é um laço e representa **retenção** (atendimento
na própria região). A soma dos laços sobre o total de saídas de uma região dá o
Índice de Retenção Regional (IRR).

### `(Estabelecimento)-[:LOCALIZADO_EM]->(Municipio)`
Onde fica o estabelecimento. Sem propriedades.

### `(Estabelecimento)-[:REALIZA]->(Procedimento)`
Oferta efetiva: o estabelecimento produziu aquele procedimento.

| Propriedade | Tipo | Descrição |
|---|---|---|
| `ano` | int | Ano de competência |
| `volume` | int | Número de atendimentos produzidos |

## Diagrama

```
        PERTENCE_A                 FLUXO {ano, complexidade, linha, volume, deslocou}
Municipio ────────► RegiaoSaude ◄──────────────────────────────────► RegiaoSaude
    ▲                                                                      │ (origem = residencia,
    │ LOCALIZADO_EM                                                        │  destino = atendimento)
    │                                                                      │
Estabelecimento ──REALIZA {ano, volume}──► Procedimento {grupo 0304, cid_grupo, complexidade}
```

## Restrições e índices (Cypher)

```cypher
CREATE CONSTRAINT regiao_codigo   IF NOT EXISTS FOR (r:RegiaoSaude)    REQUIRE r.codigo IS UNIQUE;
CREATE CONSTRAINT municipio_codigo IF NOT EXISTS FOR (m:Municipio)     REQUIRE m.codigo IS UNIQUE;
CREATE CONSTRAINT estab_cnes      IF NOT EXISTS FOR (e:Estabelecimento) REQUIRE e.cnes  IS UNIQUE;
CREATE CONSTRAINT proc_codigo     IF NOT EXISTS FOR (p:Procedimento)   REQUIRE p.codigo IS UNIQUE;
```

## Perguntas de exemplo e o Cypher esperado

Estas perguntas guiam o conjunto de avaliação (`eval/perguntas.yaml`) e servem de
exemplos para o Text2Cypher.

**1. Quais regiões mais enviam pacientes de câncer para fora?**
```cypher
MATCH (o:RegiaoSaude)-[f:FLUXO]->(d:RegiaoSaude)
WHERE f.deslocou = true
RETURN o.nome AS regiao, sum(f.volume) AS evasao
ORDER BY evasao DESC LIMIT 10;
```

**2. Qual o polo que mais recebe pacientes de fora para radioterapia?**
```cypher
MATCH (o:RegiaoSaude)-[f:FLUXO]->(d:RegiaoSaude)
WHERE f.deslocou = true AND f.linha = 'radioterapia'
RETURN d.nome AS polo, sum(f.volume) AS invasao
ORDER BY invasao DESC LIMIT 5;
```

**3. Qual o Índice de Retenção Regional da região X na alta complexidade?**
```cypher
MATCH (o:RegiaoSaude {nome: 'X'})-[f:FLUXO]->(:RegiaoSaude)
WHERE f.complexidade = 'alta'
WITH sum(CASE WHEN f.deslocou = false THEN f.volume ELSE 0 END) AS retido,
     sum(f.volume) AS total
RETURN retido * 1.0 / total AS irr;
```

**4. Quais municípios da região X ficam mais longe do polo?**
```cypher
MATCH (m:Municipio)-[:PERTENCE_A]->(r:RegiaoSaude {nome: 'X'})
RETURN m.nome AS municipio, m.dist_polo_km AS distancia
ORDER BY distancia DESC LIMIT 10;
```
