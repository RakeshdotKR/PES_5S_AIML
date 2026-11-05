# workshop\Day4\P2.py
import json
import argparse
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from dotenv import load_dotenv

# LangGraph core — Session 1 uses StateGraph/START/END/add_messages/ToolNode.
# MemorySaver + interrupt/Command are Session 2's; imported now so this header
# never needs editing between sessions.
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver      
from langgraph.types import interrupt, Command           

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI   # cloud free-tier brain
from langchain_ollama import ChatOllama                     # local free-tier brain

import sys
from pathlib import Path

# 1. Get the workspace root and Day3 path explicitly
project_root = Path(__file__).resolve().parent.parent
day3_dir = project_root / "Day3"

# 2. Add Day3 first so Day3 scripts prioritize Day3 files, then add project root
sys.path.insert(0, str(day3_dir))
sys.path.append(str(project_root))

# 3. Day 3's engine — the search tool literally calls this function.
from Day3.D3P3 import answer as rag_answer

load_dotenv()


# ── Free-tier brain configuration (same models as Days 1-3) ──────────────────
GEMINI_MODEL = "gemini-3.6-flash"   # cloud, free. Out of daily quota? Try "gemini-3.8-flash" (also free, may 503).
OLLAMA_MODEL = "gemma2:4b"          # local, 8 GB RAM friendly (fallback: gemma3:1b)

MAX_ITERATIONS = 5                  # kill-switch — Day 5 turns this into a full guardrail (budget + allow-list)


def msg_text(message: Any) -> str:
    """
    Extract readable plain text from a LangChain message.

    Why this exists: Gemini responses sometimes return `content` as a LIST of
    typed blocks ( [{'type': 'text', 'text': '...'}] ) instead of a plain
    string — printing that dumps Python repr + signatures into the terminal.
    Handles str, list-of-blocks (dict or object form), and anything else.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                # Only text blocks; skip thinking/signature-style blocks.
                if block.get("type") in (None, "text"):
                    parts.append(block.get("text", ""))
            else:  # object-style block with a .text attribute
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def make_llm(brain: str = "gemini"):
    """
    One factory for both free-tier brains — used by every LLM call today
    (agent node, orchestrator, synthesizer) so a brain swap is ONE argument.

    NOTE: deliberately NO temperature=. gemini-3.6-flash uses fixed sampling
    defaults; passing temperature only produces an SDK warning.
    """
    if brain == "gemini":
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    if brain == "ollama":
        return ChatOllama(model=OLLAMA_MODEL)
    raise ValueError(f"Unknown brain '{brain}'. Choose: gemini | ollama")

# Execution
# python -c "import P2; print('header OK')"

