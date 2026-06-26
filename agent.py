"""LangChain tool-calling agent behind /copilot.

The agent decides which dataset tools to call to answer a free-text question.
Inspecting the message trace exposes which datasets it read (great demo signal).
The deterministic maths lives in scoring.py — the LLM only routes and explains.

Built on the langchain 1.x agent API (``create_agent``, a compiled LangGraph
agent). The LLM is served via OpenRouter (OpenAI-compatible API). Configure with:
  OPENROUTER_API_KEY   (required to enable /copilot)
  OPENROUTER_MODEL     (default: anthropic/claude-sonnet-4.5)
  OPENROUTER_BASE_URL  (default: https://openrouter.ai/api/v1)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv

from context import COLUMN_DOCS, SYSTEM_PROMPT
from tools import ALL_TOOLS

load_dotenv()

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@lru_cache(maxsize=1)
def _get_agent():
    """Build the agent lazily so the API can boot without a key set."""
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        temperature=0,
        max_tokens=1500,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    )

    return create_agent(
        llm,
        ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT + "\n\nCOLUMN REFERENCE:\n" + COLUMN_DOCS,
    )


def ask(question: str) -> Dict[str, Any]:
    """Answer a free-text question via the tool-calling agent."""
    if not os.getenv("OPENROUTER_API_KEY"):
        return {
            "answer": "OPENROUTER_API_KEY is not set. Add it to backend/.env to enable the copilot.",
            "tools_used": [],
            "evidence": [],
        }

    from langchain_core.messages import AIMessage, ToolMessage

    agent = _get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])

    tools_used = [m.name for m in messages if isinstance(m, ToolMessage)]
    evidence = [m.content for m in messages if isinstance(m, ToolMessage)]

    answer = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            answer = m.content
            break

    return {"answer": answer, "tools_used": tools_used, "evidence": evidence}
