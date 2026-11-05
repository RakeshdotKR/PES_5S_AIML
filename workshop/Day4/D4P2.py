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
OLLAMA_MODEL = "gemma2:2b"          # local, 8 GB RAM friendly (fallback: gemma3:1b)

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

# ***************************************************************************************************************
@tool
def search_knowledge_base(question: str) -> Dict[str, Any]:
    """
    Search the campus handbook knowledge base and return an answer with citations.
    Use this for ANY fact about the college: library rules, fees, labs, attendance,
    scholarships, hostel, exams. The result contains 'abstained=true' when the
    handbook does not cover the topic — in that case DO NOT search again; tell
    the user the information is not in the knowledge base.
    """
    try:
        # Day 3 engine, unchanged. The INNER generator stays gemini-3.6-flash
        # even when the OUTER brain is swapped — this isolates DECISION quality
        # in Session 2's brain-swap study (only the brain choosing tools moves).
        result = rag_answer(question, backend="chroma", llm="gemini", topk=5)
        return {
            "answer":    result["text"],
            "citations": result["citations"],
            "abstained": result["abstained"],
        }
    except Exception as e:
        # Errors become JSON OBSERVATIONS, not exceptions — the model reads
        # them and can self-correct. (Day 5 makes this a guardrail rule.)
        return {"error": f"search failed: {e}"}


@tool
def calculator(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression and return the numeric result. Use this
    whenever arithmetic is needed over numbers found in tool results (fees,
    fines, totals, percentages). Supports +, -, *, /, ** and parentheses.
    Numeric expression only — no units, no words. Example: '12 * 5'.
    """
    try:
        # Same safe-eval allowlist as Days 1/3: empty __builtins__ => the
        # string cannot import modules or touch attributes. Never eval
        # untrusted strings without this pattern.
        value = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
        return {"result": round(float(value), 6)}
    except Exception as e:
        return {"error": f"could not evaluate '{expression}': {e}"}


# The tool list the model sees (order = preference hint for small models).
AGENT_TOOLS = [search_knowledge_base, calculator]

# System instruction: role + citation habit + the abstain-not-guess rule.
# (Same contract as naive_agent.py — behavior must match.)
SYSTEM_PROMPT = """You are a campus knowledge agent.
- For any fact about the college, call search_knowledge_base.
- For arithmetic over retrieved numbers, call calculator.
- Call ONE tool at a time so results stay easy to follow.
- If a search returns abstained=true, do not retry the search — answer that the
  information is not in the knowledge base.
- Cite handbook facts using the citation markers returned by the search tool.
- Never invent handbook rules, fees, or dates."""

class AgentState(TypedDict):
    """
    The state that flows through the agent graph — the three fields the
    curriculum requires, each with a job:
      messages         full conversation history (reducer: append)
      tool_call_count  cost/observability — tool calls spent this run
      iterations       loop guard — agent-node executions (cap = MAX_ITERATIONS)
    """
    messages: Annotated[list, add_messages]
    tool_call_count: int
    iterations: int


def build_agent_graph(brain: str = "gemini"):
    r"""
    (different brains) without editing code. Session 2 extends this signature
    with require_approval / auto_approve / checkpointer — the node logic below
    does not change.

    Graph shape (the naive_agent.py while-loop, drawn):

        START -> agent --(tool calls?)--> tools --> agent --(tool calls?)--> ...
                       \--(final text / cap reached)--> END
    """
    # Bind tools ONCE per compiled graph (binding is cheap, but doing it once
    # mirrors how production agents configure their model at startup).
    llm_with_tools = make_llm(brain).bind_tools(AGENT_TOOLS)

    def agent_node(state: AgentState) -> dict:
        """One LLM call. Returns a partial update: the model's message plus
        the two counters. System message is prepended here (not stored in
        state) so history stays user/assistant/tool only."""
        response = llm_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {
            "messages": [response],
            # Increment by the number of calls this response made (usually 0 or 1).
            "tool_call_count": state.get("tool_call_count", 0) + len(response.tool_calls),
            "iterations": state.get("iterations", 0) + 1,
        }

    def should_continue(state: AgentState) -> str:
        """
        THE conditional edge — 'did the model request a tool?'

        The prebuilt `tools_condition` (from langgraph.prebuilt) returns
        exactly "tools" or "__end__" for this same check. We write our own
        because we ALSO need the iteration kill-switch — the Day 5 guardrail
        preview: no runaway loops, ever, regardless of what the model wants.
        """
        if state.get("iterations", 0) >= MAX_ITERATIONS:
            return "__end__"                       # cap hit: stop, even mid-tooling
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):      # model asked for a tool
            return "tools"
        return "__end__"                           # final text: done

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)          # the LLM decision step
    builder.add_node("tools", ToolNode(AGENT_TOOLS))  # executes tools, emits ToolMessages
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        # path_map: what each return value means (name -> node, "__end__" -> END)
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "agent")             # THE LOOP: tools -> agent -> ...

    # Session 1: no checkpointer — every invoke starts from scratch (stateless,
    # exactly like naive_agent.py). Session 2 passes MemorySaver() here.
    return builder.compile()

# Execution: python -c "import D4P2; g = D4P2.build_agent_graph(); print('graph built:', [n for n in g.get_graph().nodes])"
# python -c "import D4P2; print(D4P2.make_llm('ollama').invoke('Reply with: Ollama connection successful'))"
# python -c "import D4P2; print(D4P2.search_knowledge_base.invoke({'question':'What are the library timings?'}))"
