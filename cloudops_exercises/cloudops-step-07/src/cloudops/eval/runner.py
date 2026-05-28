"""Shared eval runtime helpers used by CLI and tests."""
from __future__ import annotations

from cloudops.observability.opik import build_langchain_config


def run_question(agent, *, question: str, run_name: str, session_id: str, step_tag: str = "cloudops-step-07") -> str:
    """Invoke the agent and return the final text response."""
    config = build_langchain_config(
        run_name=run_name,
        session_id=session_id,
        tags=[step_tag, "eval", "deepeval"],
        metadata={"entrypoint": "cloudops-eval", "question_hash": str(hash(question))},
        configurable={"thread_id": f"eval-{hash(question)}"},
    )
    final = ""
    for chunk in agent.stream(
        {"messages": [("user", question)]},
        config=config,
        stream_mode="updates",
    ):
        for node_output in chunk.values():
            if not isinstance(node_output, dict):
                continue
            msgs = node_output.get("messages", [])
            for msg in msgs:
                if hasattr(msg, "content") and msg.content:
                    final = msg.content if isinstance(msg.content, str) else str(msg.content)
    return final
