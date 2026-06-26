"""LangChain tool-calling agent behind /copilot.

The agent decides which dataset tools to call to answer a free-text question.
``return_intermediate_steps=True`` exposes which datasets it read (great demo
signal). The deterministic maths lives in scoring.py — the LLM only routes and
explains.

The LLM is served via OpenRouter (OpenAI-compatible API). Configure with:
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
def _get_executor():
    """Build the agent executor lazily so the API can boot without a key set."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        temperature=0,
        max_tokens=1500,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT + "\n\nCOLUMN REFERENCE:\n" + COLUMN_DOCS),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        return_intermediate_steps=True,
        max_iterations=6,
        verbose=False,
    )


def ask(question: str) -> Dict[str, Any]:
    """Answer a free-text question via the tool-calling agent."""
    if not os.getenv("OPENROUTER_API_KEY"):
        return {
            "answer": "OPENROUTER_API_KEY is not set. Add it to backend/.env to enable the copilot.",
            "tools_used": [],
            "evidence": [],
        }

    executor = _get_executor()
    result = executor.invoke({"input": question})
    return {
        "answer": result["output"],
        "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])],
        "evidence": [step[1] for step in result.get("intermediate_steps", [])],
    }
