import sys

from langchain_core.messages import AIMessage
from rich.console import Console
from rich.markdown import Markdown

from cloudops.agents.react import build_agent

console = Console()


def _extract_answer(stream_chunks: list) -> str:
    """Pull the last non-empty AI message content from stream chunks."""
    for chunk in reversed(stream_chunks):
        for update in chunk.values():
            for msg in reversed(update.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content
    return "(no response)"


def main() -> None:
    console.print("[bold blue]CloudOps Analyst[/bold blue] — Step 1: ReAct + CSV tools")
    console.print("[dim]Type your question or 'exit' to quit.[/dim]\n")

    agent = build_agent()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            sys.exit(0)

        if not query or query.lower() in ("exit", "quit"):
            sys.exit(0)

        console.print("[dim]thinking...[/dim]", end="\r")

        # "updates" yields per-node state diffs, so tool calls appear as they happen
        chunks: list = []
        for chunk in agent.stream(
            {"messages": [("user", query)]},
            stream_mode="updates",
        ):
            chunks.append(chunk)
            for node, update in chunk.items():
                for msg in update.get("messages", []):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                            console.print(f"  [dim]→ {tc['name']}({args})[/dim]")

        answer = _extract_answer(chunks)
        console.print(f"\n[green]Agent:[/green]")
        console.print(Markdown(answer))
        console.print()


if __name__ == "__main__":
    main()
