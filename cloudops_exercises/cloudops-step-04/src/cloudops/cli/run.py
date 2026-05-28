import sys
import uuid

from langchain_core.messages import AIMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from cloudops.agents.react import build_agent

console = Console()


def _extract_answer(stream_chunks: list) -> str:
    for chunk in reversed(stream_chunks):
        for update in chunk.values():
            for msg in reversed(update.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content
    return "(no response)"


def _run_turn(agent, user_input: str | None, config: dict, resume_value: str | None = None) -> str:
    """Stream one agent turn; handle any HITL interrupts inline.

    Returns the final answer string after all interrupts are resolved.
    """
    chunks: list = []

    invocation = (
        Command(resume=resume_value)
        if resume_value is not None
        # each new user message starts a fresh turn on the same thread
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
        # GraphInterrupt is raised when interrupt() is called inside a tool.
        # Inspect the payload, prompt the human, then resume.
        cls_name = type(exc).__name__
        if cls_name == "GraphInterrupt":
            # exc.args[0] is a tuple of Interrupt objects; each has a .value dict
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

                # resume the graph with the human's answer
                return _run_turn(agent, user_input=None, config=config, resume_value=human_answer)

        raise  # re-raise unexpected exceptions

    return _extract_answer(chunks)


def main() -> None:
    console.print("[bold blue]CloudOps Analyst[/bold blue] — Step 4: ReAct + HITL + Eval Harness")
    console.print("[dim]Type your question or 'exit' to quit.[/dim]\n")

    agent = build_agent()
    # one thread_id per CLI session — MemorySaver keeps history in-process
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            sys.exit(0)

        if not query or query.lower() in ("exit", "quit"):
            sys.exit(0)

        console.print("[dim]thinking...[/dim]", end="\r")

        answer = _run_turn(agent, user_input=query, config=config)
        console.print(f"\n[green]Agent:[/green]")
        console.print(Markdown(answer))
        console.print()


if __name__ == "__main__":
    main()
