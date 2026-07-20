"""
llm.py — cliente LLM com rotacao de chaves, fallback de modelo, saida
estruturada e contabilidade de custo.

Reaproveita os padroes do pipeline de revisao AMR/EMERALD (extracao de artigos
com Gemini): uma chave por variavel numerada, rotacao round-robin para respeitar
o rate limit do free tier, cadeia de modelos de fallback no retry, cliente
agnostico de SDK (google-genai novo -> google-generativeai antigo) e extracao de
uso de tokens para controlar custo.

Provedores:
  - gemini (primario): usa GEMINI_KEY_1..N e GEMINI_MODEL
  - groq   (alternativo): usa GROQ_KEY_1..N e GROQ_MODEL (API compativel OpenAI)
  - mistral (alternativo): usa MISTRAL_KEY_1..N e MISTRAL_MODEL (ex.: codestral-latest;
                    especialista em codigo; API compativel OpenAI)
  - ollama (local): modelo aberto servido pelo Ollama, sem chave nem rate limit
                    (OLLAMA_MODEL, OLLAMA_BASE_URL; API compativel OpenAI)

Uso:
    from src.llm import LLM
    llm = LLM(provider="gemini")
    texto = llm.gerar(prompt)               # texto livre
    obj   = llm.gerar_json(prompt)          # dict validado (JSON)
    print(llm.custo())                      # {'chamadas', 'prompt_tokens', ...}
"""

import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

# Cadeia de fallback do Gemini (rotaciona no retry, como no pipeline AMR).
# gemini-1.5-flash foi aposentado (retorna 404) e nao entra mais na cadeia.
GEMINI_FALLBACKS = ["gemini-2.0-flash", "gemini-2.5-flash"]
MAX_RETRIES = 3
RETRY_DELAY = 8       # segundos entre tentativas
CALL_DELAY = 1        # segundos entre chamadas (rate limiting suave)
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 2048


def _carregar_chaves(prefixo):
    """Le CHAVE_1, CHAVE_2, ... do ambiente. Aceita tambem uma lista
    separada por virgula em CHAVE (fallback)."""
    chaves = []
    i = 1
    while True:
        v = os.getenv(f"{prefixo}_{i}")
        if v is None:
            break
        if v.strip():
            chaves.append(v.strip())
        i += 1
    if not chaves:
        bruto = os.getenv(prefixo, "")
        chaves = [k.strip() for k in bruto.split(",") if k.strip()]
    return chaves


class LLM:
    def __init__(self, provider="gemini", model=None, max_output_tokens=MAX_OUTPUT_TOKENS):
        self.provider = provider
        self.max_output_tokens = max_output_tokens
        self._i = 0
        self._uso = {"chamadas": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        self.base_url = None
        if provider == "gemini":
            self.chaves = _carregar_chaves("GEMINI_KEY")
            self.model = model or os.getenv("GEMINI_MODEL", GEMINI_FALLBACKS[0])
            self.modelos = [self.model] + [m for m in GEMINI_FALLBACKS if m != self.model]
        elif provider == "groq":
            self.chaves = _carregar_chaves("GROQ_KEY")
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self.modelos = [self.model]
            self.base_url = "https://api.groq.com/openai/v1"
        elif provider == "mistral":
            self.chaves = _carregar_chaves("MISTRAL_KEY")
            self.model = model or os.getenv("MISTRAL_MODEL", "codestral-latest")
            self.modelos = [self.model]
            self.base_url = "https://api.mistral.ai/v1"
        elif provider == "ollama":
            # modelo aberto servido localmente pelo Ollama; sem chave, sem rate
            # limit. API compativel com OpenAI. Rode `ollama serve` + pull do modelo.
            self.chaves = ["ollama"]  # placeholder: Ollama nao exige chave
            self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
            self.modelos = [self.model]
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        else:
            raise ValueError(f"Provedor desconhecido: {provider}")

        if not self.chaves:
            raise RuntimeError(
                f"Nenhuma chave para o provedor '{provider}'. "
                f"Preencha {provider.upper()}_KEY_1 no .env (veja .env.example)."
            )

    # ── rotacao round-robin de chaves ──
    def _proxima_chave(self):
        chave = self.chaves[self._i % len(self.chaves)]
        self._i += 1
        return chave

    # ── contabilidade de custo ──
    def custo(self):
        return dict(self._uso)

    def _somar_uso(self, prompt_tokens, output_tokens):
        self._uso["chamadas"] += 1
        self._uso["prompt_tokens"] += prompt_tokens or 0
        self._uso["output_tokens"] += output_tokens or 0
        self._uso["total_tokens"] += (prompt_tokens or 0) + (output_tokens or 0)

    # ── chamada de baixo nivel, com retry + fallback de modelo ──
    def gerar(self, prompt, system=None):
        """Retorna texto. Rotaciona chave e modelo a cada tentativa."""
        ultimo_erro = None
        for tentativa in range(MAX_RETRIES):
            chave = self._proxima_chave()
            modelo = self.modelos[tentativa % len(self.modelos)]
            try:
                if self.provider == "gemini":
                    texto, pt, ot = self._chamar_gemini(chave, modelo, prompt, system)
                else:
                    texto, pt, ot = self._chamar_openai_compat(chave, modelo, prompt, system)
                self._somar_uso(pt, ot)
                time.sleep(CALL_DELAY)
                return texto
            except Exception as e:  # noqa: BLE001 — queremos tentar de novo
                ultimo_erro = e
                if tentativa < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        raise RuntimeError(f"LLM falhou apos {MAX_RETRIES} tentativas: {ultimo_erro}")

    def gerar_json(self, prompt, system=None):
        """Retorna dict. Reforca 'so JSON' e faz parsing tolerante a cercas."""
        reforco = prompt + "\n\nResponda APENAS com JSON valido, sem texto fora do objeto."
        return _extrair_json(self.gerar(reforco, system=system))

    # ── Gemini (SDK novo -> SDK antigo) ──
    def _chamar_gemini(self, chave, modelo, prompt, system):
        conteudo = prompt if system is None else f"{system}\n\n{prompt}"
        # SDK novo: google-genai
        try:
            from google import genai as google_genai

            client = google_genai.Client(api_key=chave)
            resp = client.models.generate_content(
                model=modelo,
                contents=[conteudo],
                config={"max_output_tokens": self.max_output_tokens, "temperature": TEMPERATURE},
            )
            pt, ot = self._uso_gemini(resp)
            return resp.text.strip(), pt, ot
        except ImportError:
            pass
        # SDK antigo: google-generativeai
        import google.generativeai as genai

        genai.configure(api_key=chave)
        gm = genai.GenerativeModel(modelo)
        resp = gm.generate_content(
            [conteudo],
            generation_config={"temperature": TEMPERATURE, "max_output_tokens": self.max_output_tokens},
        )
        pt, ot = self._uso_gemini(resp)
        return resp.text.strip(), pt, ot

    @staticmethod
    def _uso_gemini(resp):
        try:
            um = resp.usage_metadata
            return (
                getattr(um, "prompt_token_count", None) or um.get("prompt_token_count"),
                getattr(um, "candidates_token_count", None) or um.get("candidates_token_count"),
            )
        except Exception:  # noqa: BLE001
            return None, None

    # ── Groq / Ollama (API compativel com OpenAI) ──
    def _chamar_openai_compat(self, chave, modelo, prompt, system):
        from openai import OpenAI

        client = OpenAI(api_key=chave, base_url=self.base_url)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=modelo, messages=msgs, temperature=TEMPERATURE, max_tokens=self.max_output_tokens
        )
        u = resp.usage
        return resp.choices[0].message.content.strip(), u.prompt_tokens, u.completion_tokens


def _extrair_json(bruto):
    """Extrai um objeto JSON de uma resposta de LLM (tolerante a cercas markdown)."""
    limpo = re.sub(r"^```json\s*", "", bruto)
    limpo = re.sub(r"^```\s*", "", limpo)
    limpo = re.sub(r"\s*```$", "", limpo).strip()
    try:
        return json.loads(limpo)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", limpo, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Nao consegui extrair JSON da resposta: {bruto[:200]}")
