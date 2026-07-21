"""
agente.py — GraphRAG agentico sobre o grafo do SUS (orquestrado com LangGraph).

Envolve o Text2Cypher num agente de estados com quatro capacidades que um
pipeline de disparo unico nao tem:

  1. Auto-correcao   — se o Cypher gerado falha no EXPLAIN (ou na execucao),
                       o erro volta ao prompt e o modelo tenta de novo, ate
                       MAX_TENTATIVAS. (padrao do ai-voice-agent)
  2. Timeout         — cada execucao no Neo4j tem limite de tempo, para nao
                       travar em consulta pesada ou modelo local lento.
                       (padrao do Multi-Agent-Task-Solver)
  3. Human-in-the-loop — antes de executar, um gate pausa o grafo (interrupt do
                       LangGraph) e pede aprovar / editar / rejeitar o Cypher.
                       Em saude, nada roda sem revisao. (padrao do agent_4_social_media)
  4. Sintese         — as linhas do grafo viram uma resposta em portugues: o
                       "G" (generation) que fecha o ciclo do GraphRAG.

Reaproveita montar_prompt/LLM (geracao), EXPLAIN e executar dos modulos
existentes; a orquestracao (nos, arestas condicionais, estado, interrupt) e o
que este arquivo adiciona.

Grafo de estados:

    START -> gerar -> validar --valido----------> gate --aprovar--> executar --ok--> sintetizar -> END
                        |  ^                        |  \\--rejeitar--> rejeitado -> END       |
                        |  |  (erro realimenta o prompt)                                      |
                        \\--invalido, tentativas<max--/ <----------- erro em runtime ---------/
                           invalido, esgotou -> falha -> END

Uso:
    python src/agente.py "Quais regioes mais evadem pacientes de cancer?"
    python src/agente.py "..." --provider ollama --auto      # sem gate humano
    python src/agente.py "..." --max-tentativas 4
"""

import argparse
import concurrent.futures
import os
import sys
from typing import Optional, TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

from esquema import montar_prompt  # noqa: E402
from llm import LLM  # noqa: E402
from text2cypher import gerar_local, imprimir_tabela, limpar_cypher  # noqa: E402

load_dotenv()

MAX_TENTATIVAS = 3       # geracoes por pergunta (a 1a + as auto-correcoes)
TIMEOUT_EXEC = 30        # segundos por execucao no Neo4j
LIMITE_LINHAS = 50       # teto de linhas trazidas do grafo


# ── conexao Neo4j (driver unico, reaproveitado pelos nos) ──
_DRIVER = {}


def _driver():
    if "d" not in _DRIVER:
        from neo4j import GraphDatabase

        _DRIVER["d"] = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
        )
    return _DRIVER["d"]


# ── estado do agente (o que trafega entre os nos) ──
class Estado(TypedDict, total=False):
    pergunta: str
    provider: str
    auto: bool
    max_tentativas: int
    cypher: Optional[str]
    erro: Optional[str]
    tentativas: int
    linhas: Optional[list]
    resposta: Optional[str]
    decisao: Optional[str]
    historico: list
    tokens: int
    status: str


# ── nos ──
def _bloco_correcao(cypher_ruim, erro):
    """Realimentacao: descreve a falha anterior para o modelo nao repeti-la."""
    return (
        "\n\nA tentativa anterior FALHOU. Nao repita o mesmo erro.\n"
        f"Cypher anterior:\n{cypher_ruim}\n"
        f"Erro do Neo4j:\n{erro}\n"
        "Gere uma consulta Cypher corrigida, usando apenas o esquema acima."
    )


def gerar_cypher(estado: Estado) -> Estado:
    """Gera o Cypher. Na 2a tentativa em diante, injeta o erro anterior no prompt."""
    pergunta = estado["pergunta"]
    provider = estado.get("provider", "gemini")
    erro = estado.get("erro")
    cypher_ruim = estado.get("cypher")
    tentativas = estado.get("tentativas", 0) + 1
    tokens = estado.get("tokens", 0)

    if provider == "local":
        # o modelo fine-tunado usa prompt fixo (sem realimentacao de erro)
        cypher, custo = gerar_local(pergunta)
    else:
        prompt = montar_prompt(pergunta)
        if erro:
            prompt += _bloco_correcao(cypher_ruim, erro)
        llm = LLM(provider=provider)
        cypher = limpar_cypher(llm.gerar(prompt))
        custo = llm.custo()
    if custo:
        tokens += custo.get("total_tokens", 0)

    hist = list(estado.get("historico", []))
    hist.append({"tentativa": tentativas, "cypher": cypher, "corrigindo": bool(erro)})
    return {"cypher": cypher, "tentativas": tentativas, "tokens": tokens, "historico": hist, "erro": None}


def validar(estado: Estado) -> Estado:
    """EXPLAIN: valida sintaxe e schema sem executar. Captura a mensagem de erro."""
    cypher = estado["cypher"]
    try:
        with _driver().session() as s:
            s.run(f"EXPLAIN {cypher}").consume()
        return {"erro": None}
    except Exception as e:  # noqa: BLE001
        return {"erro": str(e).strip().splitlines()[0][:300]}


def gate_humano(estado: Estado) -> Estado:
    """Human-in-the-loop: pausa o grafo e espera a decisao (a menos que --auto)."""
    if estado.get("auto"):
        return {"decisao": "aprovar"}
    resp = interrupt(
        {
            "tipo": "revisao_cypher",
            "pergunta": estado["pergunta"],
            "cypher": estado["cypher"],
            "tentativa": estado.get("tentativas"),
        }
    )
    # resp: string ("aprovar"/"rejeitar") ou dict {"acao": ..., "cypher": ...}
    if isinstance(resp, dict):
        acao = resp.get("acao", "aprovar")
        if acao == "editar" and resp.get("cypher"):
            return {"cypher": limpar_cypher(resp["cypher"]), "decisao": "aprovar"}
        return {"decisao": acao}
    return {"decisao": resp or "aprovar"}


def _rodar(cypher):
    with _driver().session() as s:
        return [dict(r) for r in s.run(cypher)][:LIMITE_LINHAS]


def executar(estado: Estado) -> Estado:
    """Executa no Neo4j com timeout. Erro de runtime tambem alimenta a auto-correcao."""
    cypher = estado["cypher"]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            linhas = ex.submit(_rodar, cypher).result(timeout=TIMEOUT_EXEC)
        return {"linhas": linhas, "erro": None}
    except concurrent.futures.TimeoutError:
        return {"erro": f"execucao excedeu {TIMEOUT_EXEC}s"}
    except Exception as e:  # noqa: BLE001
        return {"erro": str(e).strip().splitlines()[0][:300]}


def sintetizar(estado: Estado) -> Estado:
    """Transforma as linhas do grafo em resposta em portugues (o 'G' do GraphRAG)."""
    linhas = estado.get("linhas") or []
    provider = estado.get("provider", "gemini")
    if not linhas:
        return {"resposta": "A consulta nao retornou resultados.", "status": "ok"}
    if provider == "local":
        # modelo pequeno: nao sintetiza, entrega a tabela crua
        return {"resposta": None, "status": "ok"}
    prompt = (
        "Responda a pergunta do usuario em portugues, de forma objetiva, usando "
        "SOMENTE os dados da tabela. Nao invente numeros nem nomes.\n\n"
        f"PERGUNTA: {estado['pergunta']}\n\nDADOS (JSON):\n{linhas[:20]}\n\nRESPOSTA:"
    )
    llm = LLM(provider=provider)
    resposta = llm.gerar(prompt).strip()
    tokens = estado.get("tokens", 0) + (llm.custo().get("total_tokens", 0) or 0)
    return {"resposta": resposta, "tokens": tokens, "status": "ok"}


def falha(estado: Estado) -> Estado:
    return {"status": "falha"}


def rejeitado(estado: Estado) -> Estado:
    return {"status": "rejeitado"}


# ── roteadores (arestas condicionais) ──
def roteia_validacao(estado: Estado) -> str:
    if not estado.get("erro"):
        return "gate"
    if estado.get("tentativas", 0) < estado.get("max_tentativas", MAX_TENTATIVAS):
        return "gerar"
    return "falha"


def roteia_gate(estado: Estado) -> str:
    return "executar" if estado.get("decisao") == "aprovar" else "rejeitado"


def roteia_execucao(estado: Estado) -> str:
    if not estado.get("erro"):
        return "sintetizar"
    if estado.get("tentativas", 0) < estado.get("max_tentativas", MAX_TENTATIVAS):
        return "gerar"
    return "falha"


# ── montagem do grafo ──
def construir():
    g = StateGraph(Estado)
    g.add_node("gerar", gerar_cypher)
    g.add_node("validar", validar)
    g.add_node("gate", gate_humano)
    g.add_node("executar", executar)
    g.add_node("sintetizar", sintetizar)
    g.add_node("falha", falha)
    g.add_node("rejeitado", rejeitado)

    g.add_edge(START, "gerar")
    g.add_edge("gerar", "validar")
    g.add_conditional_edges("validar", roteia_validacao, {"gate": "gate", "gerar": "gerar", "falha": "falha"})
    g.add_conditional_edges("gate", roteia_gate, {"executar": "executar", "rejeitado": "rejeitado"})
    g.add_conditional_edges("executar", roteia_execucao, {"sintetizar": "sintetizar", "gerar": "gerar", "falha": "falha"})
    g.add_edge("sintetizar", END)
    g.add_edge("falha", END)
    g.add_edge("rejeitado", END)

    return g.compile(checkpointer=MemorySaver())


# ── driver de linha de comando (trata o interrupt do gate humano) ──
def _perguntar_humano(payload):
    print(f"\n  Pergunta : {payload['pergunta']}")
    print(f"  Cypher   : {payload['cypher']}   (tentativa {payload.get('tentativa')})")
    escolha = input("\n  [a]provar / [e]ditar / [r]ejeitar? ").strip().lower()
    if escolha.startswith("e"):
        return {"acao": "editar", "cypher": input("  Novo Cypher: ").strip()}
    if escolha.startswith("r"):
        return {"acao": "rejeitar"}
    return {"acao": "aprovar"}


def _relatar(estado):
    print("\n" + "=" * 64)
    print(f"Pergunta   : {estado.get('pergunta')}")
    print(f"Cypher     : {estado.get('cypher')}")
    print(f"Tentativas : {estado.get('tentativas')}   Tokens: {estado.get('tokens')}   Status: {estado.get('status')}")
    if estado.get("status") == "rejeitado":
        print("\nRejeitado pelo revisor. Nada foi executado.")
        return
    if estado.get("status") == "falha":
        print(f"\nFalhou apos as tentativas. Ultimo erro: {estado.get('erro')}")
        return
    print("\nResultado:")
    imprimir_tabela(estado.get("linhas") or [])
    if estado.get("resposta"):
        print(f"\nResposta:\n{estado['resposta']}")


def main():
    ap = argparse.ArgumentParser(description="Agente GraphRAG (LangGraph) sobre o grafo do SUS")
    ap.add_argument("pergunta", help="pergunta em portugues")
    ap.add_argument("--provider", default="gemini", choices=["gemini", "groq", "mistral", "ollama", "local"])
    ap.add_argument("--auto", action="store_true", help="pula o gate humano (executa direto)")
    ap.add_argument("--max-tentativas", type=int, default=MAX_TENTATIVAS)
    args = ap.parse_args()

    app = construir()
    config = {"configurable": {"thread_id": "cli"}}
    estado = app.invoke(
        {
            "pergunta": args.pergunta,
            "provider": args.provider,
            "auto": args.auto,
            "max_tentativas": args.max_tentativas,
            "tentativas": 0,
            "tokens": 0,
            "historico": [],
        },
        config,
    )

    # laco human-in-the-loop: enquanto o grafo pausar num interrupt, decide e retoma
    while "__interrupt__" in estado:
        resposta = _perguntar_humano(estado["__interrupt__"][0].value)
        estado = app.invoke(Command(resume=resposta), config)

    _relatar(estado)


if __name__ == "__main__":
    main()
