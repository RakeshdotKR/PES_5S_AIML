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

# *****************************************************************************
def build_multi(brain: str = "gemini"):
    """
    Assemble the orchestrator-worker graph. Nodes that need an LLM close over
    `brain` so --brain ollama swaps EVERY outer brain at once (the retrieval
    specialist's INNER engine stays gemini — isolation rule from Day 4).

    Returns a compiled graph — no checkpointer today (the capstone adds
    thread_ids for chat UX; one-shot stress runs don't need memory).
    """
    # The retrieval specialist IS Day 4's agent graph: search + calculator with
    # the iteration kill-switch, approval OFF (an orchestrator must run
    # unattended — HITL approval points live inside guardrails, Session 2).
    retrieval_specialist = build_agent_graph(brain=brain, require_approval=False)

    def orchestrator(state: MultiState) -> dict:
        """One LLM call -> strict JSON -> validated route.
        Defensive design: parse; RETRY once on junk JSON; then fall back to
        'clarify'. A wrong route wastes a whole specialist run and can produce
        a confident wrong answer — asking the user is the cheap loss."""
        question = state.get("question") or msg_text(state["messages"][-1])
        prompt = ORCHESTRATOR_PROMPT.format(question=question)
        llm = make_llm(brain)

        raw = ""
        for _ in range(2):                       # one retry on junk JSON
            raw = msg_text(llm.invoke(prompt))
            parsed = _parse_route(raw)
            if parsed:
                return {"route": parsed["route"],
                        "route_reason": parsed["reason"],
                        "messages": [AIMessage(content=raw)]}
        # Both attempts failed -> safe default (designed loss, not a bug).
        return {"route": "clarify",
                "route_reason": "orchestrator could not classify reliably",
                "messages": [AIMessage(content=raw or "{}")]}

    def route_specialist(state: MultiState) -> Send:
        """
        THE dynamic edge. A plain conditional edge picks a node; Send() picks
        the node AND passes it a payload (here: the whole state).
        Same primitive a parallel fan-out uses — just sent to one target.

        Verified behavior (langgraph 1.2.x): the child graph receives only the
        keys ITS schema declares — extra parent keys (question, route, ...) are
        filtered out, and the child's messages merge back through the shared
        add_messages reducer (matched message ids are de-duplicated, so the
        child's view never duplicates history in the parent).

        *** TWO FIXES FOUND BY ACTUALLY RUNNING THIS CODE (see teaching note
        after Step 3) — do not "simplify" them away: ***
        1. TRAILING-AI STRIP: Gemini does NOT support model prefilling — a
           request whose FINAL message is an AI message is rejected with
           "The final request turn must be a user message or a function
           response". The orchestrator's routing JSON is the last message in
           state, so it must be stripped before a specialist sees the history.
        2. NEVER KeyError: unknown/missing route asks instead of crashing.
        """
        route = state.get("route") or "clarify"   # defensive: never KeyError
        target = {
            "retrieval": "retrieval_specialist",
            "general":   "general_specialist",
            "clarify":   "clarifier",
        }.get(route, "clarifier")                 # unknown route -> ask, don't guess

        # Specialists see the CONVERSATION, not the routing chatter. Strip the
        # single trailing orchestrator AI message (no tool_calls = routing
        # JSON, not a real answer). If nothing is left, re-seed with the
        # original question so the request always ends with a user message.
        msgs = list(state.get("messages") or [])
        if (msgs and isinstance(msgs[-1], AIMessage)
                and not getattr(msgs[-1], "tool_calls", None)):
            msgs = msgs[:-1]
        if not msgs:
            msgs = [HumanMessage(content=state.get("question", ""))]
        return Send(target, {**state, "messages": msgs})

    def general_specialist(state: MultiState) -> dict:
        """
        A mini tool-calling agent INSIDE one node (max 3 LLM calls):
        calculator + get_current_time. A node is just a Python function —
        you may nest arbitrary logic, including a small loop. Used for math,
        time, and anything that needs no handbook citation.
        """
        llm = make_llm(brain).bind_tools([calculator, get_current_time])
        new_msgs: list = []                      # only what THIS node contributes
        for _ in range(3):                       # budget: 3 LLM calls max
            resp = llm.invoke(
                [SystemMessage(content="You are a math/time specialist. Use tools "
                                        "for arithmetic or dates. Answer concisely.")]
                + state["messages"] + new_msgs
            )
            new_msgs.append(resp)
            if not resp.tool_calls:
                return {"specialist_output": msg_text(resp), "messages": new_msgs}
            for tc in resp.tool_calls:
                fn = {"calculator": calculator,
                      "get_current_time": get_current_time}.get(tc["name"])
                try:
                    # @tool objects run via .invoke(args_dict) -> raw return value.
                    observation = fn.invoke(tc["args"]) if fn else {"error": "unknown tool"}
                except Exception as e:
                    observation = {"error": str(e)}   # error -> observation, not crash
                content = (observation if isinstance(observation, str)
                           else json.dumps(observation, ensure_ascii=False))
                new_msgs.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        # Budget exhausted: return a visible, honest partial result.
        return {"specialist_output": "general specialist hit its 3-call budget",
                "messages": new_msgs}

    def clarifier(state: MultiState) -> dict:
        """HITL-lite: instead of guessing, the system ASKS. The turn ends with
        a clarifying question; the user's reply arrives as a fresh invoke."""
        text = (f"I need a bit more detail: {state.get('route_reason', 'the question is ambiguous')}. "
                f"Could you rephrase or add specifics about what you need?")
        return {"specialist_output": text, "messages": [AIMessage(content=text)]}

    def synthesizer(state: MultiState) -> dict:
        """
        The 'reconcile' step: turns specialist findings into the user-facing
        final answer — citations preserved, abstentions relayed honestly.
        For the retrieval route specialist_output was never set (the nested
        sub-graph has its own state schema), so fall back to the last message,
        which IS the sub-graph's final answer.
        """
        output = state.get("specialist_output") or msg_text(state["messages"][-1])
        prompt = SYNTH_PROMPT.format(
            route=state.get("route", "?"),
            question=state.get("question", ""),
            specialist_output=output,
        )
        resp = make_llm(brain).invoke(prompt)
        return {"specialist_output": msg_text(resp), "messages": [resp]}

    # ── Wire the graph ─────────────────────────────────────────────────────
    b = StateGraph(MultiState)
    b.add_node("orchestrator", orchestrator)
    b.add_node("retrieval_specialist", retrieval_specialist)   # nested sub-graph
    b.add_node("general_specialist", general_specialist)
    b.add_node("clarifier", clarifier)
    b.add_node("synthesizer", synthesizer)

    b.add_edge(START, "orchestrator")
    # path_map must list EXACTLY the add_node names Send() can target.
    b.add_conditional_edges("orchestrator", route_specialist,
                            ["retrieval_specialist", "general_specialist", "clarifier"])
    b.add_edge("retrieval_specialist", "synthesizer")
    b.add_edge("general_specialist", "synthesizer")
    b.add_edge("clarifier", END)        # clarify = end of turn (user answers next)
    b.add_edge("synthesizer", END)
    return b.compile()

# Execution: python -c "import D5P1; g = D5P1.build_multi(); print('multi-agent graph OK')"

# *****************************************************************************************************