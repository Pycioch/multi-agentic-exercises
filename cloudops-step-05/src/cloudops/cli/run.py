"""CLI entry point — Step 5: adds input rail and pipeline routing.

Architecture:
  User input
      ↓
  validate_input()          ← security rail (blocks injection / oversized input)
      ↓
  _is_complex()             ← routing heuristic
      ↓               ↓
  ReAct agent         Pipeline (planner → extractor → visualizer)
  (step 4, HITL)      (step 5, budget-capped, model-routed)

Simple questions (lookup, count, "who was on-call") go to the ReAct agent
because it handles them in a single LLM call with HITL support.

Complex questions (correlations, trends, "plot X vs Y") go to the Pipeline
because they decompose naturally into plan → extract → visualize stages.
"""

import sys
import uuid

from langchain_core.messages import AIMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from cloudops.agents.react import build_agent
from cloudops.core.budget import DEFAULT_BUDGET
from cloudops.graph.pipeline import build_pipeline, run_pipeline
from cloudops.security.input_rails import validate_input

console = Console()

# Keywords that signal a multi-stage analytical question suited for the Pipeline.
_COMPLEX_KEYWORDS = (
    "correlat",
    "trend",
    "over time",
    "compare",
    "plot",
    "chart",
    "graph",
    " vs ",
    "versus",
    "caused",
    "before and after",
    "relationship",
    "pattern",
)


def _is_complex(query: str) -> bool:
    """Return True if the question is better served by the Pipeline than the ReAct agent.

    Heuristic: questions that ask for analysis + visualisation, causal reasoning,
    or temporal trends map to the Pipeline's planner → extractor → visualizer stages.
    Simple lookups ("who", "what", "how many") stay with the ReAct agent.
    """
    lowered = query.lower()
    return any(kw in lowered for kw in _COMPLEX_KEYWORDS)


# ── ReAct helpers (unchanged from step 4) ────────────────────────────────────

def _extract_react_answer(stream_chunks: list) -> str:
    for chunk in reversed(stream_chunks):
        for update in chunk.values():
            if not isinstance(update, dict):
                continue
            for msg in reversed(update.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content
    return "(no response)"


def _run_react_turn(agent, user_input: str | None, config: dict, resume_value: str | None = None) -> str:
    chunks: list = []
    invocation = (
        Command(resume=resume_value)
        if resume_value is not None
        else {"messages": [("user", user_input)]}
    )
    try:
        for chunk in agent.stream(invocation, config=config, stream_mode="updates"):
            chunks.append(chunk)
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                for msg in update.get("messages", []):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                            console.print(f"  [dim]→ {tc['name']}({args})[/dim]")
    except Exception as exc:
        cls_name = type(exc).__name__
        if cls_name == "GraphInterrupt":
            interrupts = exc.args[0] if exc.args else []
            question = None
            for intr in interrupts:
                payload = getattr(intr, "value", {})
                if isinstance(payload, dict) and "question" in payload:
                    question = payload["question"]
                    break
            if question:
                console.print(f"\n[yellow]Agent asks:[/yellow] {question}")
                try:
                    human_answer = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    human_answer = "skip"
                return _run_react_turn(agent, user_input=None, config=config, resume_value=human_answer)
        raise
    return _extract_react_answer(chunks)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("[bold blue]CloudOps Analyst[/bold blue] — Step 5: Pipeline + Budget + Input Rail")
    console.print("[dim]Type your question or 'exit' to quit.[/dim]\n")

    react_agent = build_agent()
    pipeline = build_pipeline(budget=DEFAULT_BUDGET)
    session_id = str(uuid.uuid4())
    react_config = {"configurable": {"thread_id": session_id}}

    while True:
        try:
            raw_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            sys.exit(0)

        if not raw_query or raw_query.lower() in ("exit", "quit"):
            sys.exit(0)

        # ── Security rail ──────────────────────────────────────────────────
        try:
            query = validate_input(raw_query)
        except ValueError as exc:
            console.print(f"[red]Blocked:[/red] {exc}")
            continue

        console.print("[dim]thinking...[/dim]", end="\r")

        # ── Route ─────────────────────────────────────────────────────────
        try:
            if _is_complex(query):
                console.print("[dim]→ Pipeline (planner → extractor → visualizer)[/dim]")
                answer = run_pipeline(pipeline, query, thread_id=str(uuid.uuid4()))
            else:
                console.print("[dim]→ ReAct agent[/dim]")
                answer = _run_react_turn(react_agent, user_input=query, config=react_config)
        except RuntimeError as exc:
            console.print(f"[red]Budget error:[/red] {exc}")
            continue
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            continue

        console.print(f"\n[green]Agent:[/green]")
        console.print(Markdown(answer))
        console.print()


if __name__ == "__main__":
    main()
