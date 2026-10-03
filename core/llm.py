"""
core/llm.py
Thin wrapper around Groq (via langchain-groq) for all "thinking" agents
(Profiler, STTM Agent, Reporter). If no GROQ_API_KEY is configured, every
function falls back to a deterministic heuristic so the whole pipeline still
runs end-to-end for demo/dev purposes -- the LLM just makes it smarter.
"""
import os
import json
import re

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_llm = None


def llm_available() -> bool:
    return bool(GROQ_API_KEY)


def get_llm():
    """Lazily construct the ChatGroq client. Returns None if no key set."""
    global _llm
    if not llm_available():
        return None
    if _llm is None:
        from langchain_groq import ChatGroq

        _llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    return _llm


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def call_llm_json(system_prompt: str, user_prompt: str, fallback: dict | list):
    """
    Calls the LLM and expects a JSON response. On any failure (no key,
    network error, bad JSON) returns `fallback` instead of raising, so the
    pipeline degrades gracefully rather than breaking.
    Returns: (result, used_llm: bool, raw_error: str|None)
    """
    llm = get_llm()
    if llm is None:
        return fallback, False, "GROQ_API_KEY not set - using heuristic fallback"

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt + "\nRespond with ONLY valid JSON, no prose, no markdown fences."),
            HumanMessage(content=user_prompt),
        ]
        resp = llm.invoke(messages)
        raw = _strip_code_fences(resp.content)
        parsed = json.loads(raw)
        return parsed, True, None
    except Exception as e:  # noqa: BLE001
        return fallback, False, f"LLM call failed, used fallback: {e}"


def call_llm_text(system_prompt: str, user_prompt: str, fallback_text: str):
    llm = get_llm()
    if llm is None:
        return fallback_text, False, "GROQ_API_KEY not set - using heuristic fallback"
    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = llm.invoke(messages)
        return resp.content.strip(), True, None
    except Exception as e:  # noqa: BLE001
        return fallback_text, False, f"LLM call failed, used fallback: {e}"
