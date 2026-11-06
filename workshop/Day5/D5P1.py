# workshop\Day5\D5P1.py
import re
import json
import argparse
from typing import TypedDict, Annotated, Optional, Any, Callable
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send               # dynamic edges WITH a payload
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Day 4's agent, imported as a library — never re-implement: brains (make_llm),
# the text extractor (msg_text), the compiled retrieval specialist
# (build_agent_graph = search + calculator loop with the kill-switch), and the
# calculator tool itself. NOTE: importing D4P2 prints a '[retriever] Ready ...'
# line — that is its known import side effect, not an error.
import sys
from pathlib import Path

# 1. Get the workspace root and Day3 path explicitly
project_root = Path(__file__).resolve().parent.parent
day4_dir = project_root / "Day4"

# 2. Add Day3 first so Day3 scripts prioritize Day3 files, then add project root
sys.path.insert(0, str(day4_dir))
sys.path.append(str(project_root))

from Day4.D4P2 import make_llm, msg_text, build_agent_graph, calculator


@tool
def get_current_time() -> str:
    """Get today's date and current time as 'YYYY-MM-DD HH:MM:SS'. Use when the
    user asks what today's date/time is, or a question needs 'today' resolved."""
    # Plain function -> string: LangChain wraps it into a schema tool from the
    # type hint (return) + this docstring (description) — same skill as Day 4.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MultiState(TypedDict):
    """
    Shared state for the whole multi-agent run — one TypedDict every node
    reads/writes (shared-state pattern; no message queues):
      messages           history (reducer: append) — also carries tool observations
      question           the ORIGINAL user question (orchestrator + synthesizer)
      route              orchestrator decision: retrieval | general | clarify
      route_reason       one-line justification (shown on clarify / in traces)
      specialist_output  final text from whichever specialist ran
    """
    messages: Annotated[list, add_messages]
    question: str
    route: str
    route_reason: str
    specialist_output: str


# ── Orchestrator prompt: strict JSON out (your Day 1 JSON-mode skill) ────────
ORCHESTRATOR_PROMPT = """You are the ROUTING layer of a campus multi-agent system.
Decide which specialist should handle the user's question. Reply with ONE json
object ONLY (no prose, no code fences):
{{"route": "retrieval" | "general" | "clarify", "reason": "<one sentence>"}}

Route rules:
- "retrieval" : any fact about THIS college/campus handbook (library, fees, labs,
                attendance, scholarships, hostel, exams, regulations). The
                retrieval specialist can cite the handbook.
- "general"   : arithmetic, dates/time, greetings, chitchat, general knowledge.
- "clarify"   : the question is ambiguous or underspecified (e.g. "How do I pay?"
                could mean fees, fines, or a donation) — you cannot pick
                confidently. Asking beats guessing.

Question: {question}

JSON:"""


# ── Synthesizer prompt: the citation-preserving reconciler ──────────────────
SYNTH_PROMPT = """You are the synthesis layer of a campus multi-agent system.
The orchestrator routed the question to the '{route}' specialist. Its findings:

{specialist_output}

Question: {question}

Rules:
1. Write the final user-facing answer from the specialist's findings only.
2. Keep handbook citations like [1] EXACTLY as the specialist returned them —
   never invent or renumber a citation.
3. If the specialist said the answer is not in the knowledge base, relay that
   honestly. Do not add facts from your own knowledge.
4. If the specialist returned a clarifying question, pass it through as-is.
5. Keep it to 2-6 sentences unless the question needs more.
"""


def _parse_route(text: str) -> Optional[dict]:
    """Extract the FIRST {...} block, json.loads it, validate the route.
    Returns None on any failure — the caller decides the fallback. Defensive
    by design: gemini-3.6-flash occasionally wraps JSON in prose or fences."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)   # DOTALL: multi-line JSON
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if data.get("route") not in ("retrieval", "general", "clarify"):
        return None
    data.setdefault("reason", "")
    return data

# Execution: python -c "import D5P1; print('step 1 OK')"