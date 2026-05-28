"""Langfuse helpers for tracing and experiment execution."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Callable

_CLIENT = None
_HANDLER = None


def _langfuse_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        raise RuntimeError("Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY.")
    from langfuse import get_client

    host = os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")).strip()
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = host
    _CLIENT = get_client()
    return _CLIENT


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse_handler(*, fresh: bool = False):
    """Return LangChain callback handler if Langfuse is configured.

    fresh=True creates a new handler instance (useful when trace IDs must map
    one-to-one with a single invocation, e.g. eval scoring).
    """
    global _HANDLER
    _langfuse_client()
    if not fresh and _HANDLER is not None:
        return _HANDLER

    from langfuse.langchain import CallbackHandler
    handler = CallbackHandler()
    if not fresh:
        _HANDLER = handler
    return handler


def build_langchain_config(
    *,
    run_name: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    configurable: Mapping[str, Any] | None = None,
    recursion_limit: int | None = None,
    fresh_handler: bool = False,
) -> dict[str, Any]:
    """Build RunnableConfig with Langfuse callback and metadata when available."""
    cfg: dict[str, Any] = {}
    if configurable:
        cfg["configurable"] = dict(configurable)
    if recursion_limit is not None:
        cfg["recursion_limit"] = recursion_limit

    handler = get_langfuse_handler(fresh=fresh_handler)
    cfg["callbacks"] = [handler]

    md = dict(metadata or {})
    md["langfuse_trace_name"] = run_name
    if session_id:
        md["langfuse_session_id"] = session_id
    if tags:
        md["langfuse_tags"] = tags
    if user_id:
        md["langfuse_user_id"] = user_id
    if md:
        cfg["metadata"] = md

    return cfg


def sync_dataset(dataset_name: str, items: list[dict], description: str = "") -> int:
    """Create/update a Langfuse dataset and ingest items.

    Returns the number of items successfully submitted.
    """
    client = _langfuse_client()
    client.create_dataset(name=dataset_name, description=description or None)

    inserted = 0
    for item in items:
        client.create_dataset_item(
            dataset_name=dataset_name,
            input=item["input"],
            expected_output=item.get("expected_output"),
            metadata=item.get("metadata"),
        )
        inserted += 1
    client.flush()
    return inserted


def run_dataset_experiment(
    *,
    dataset_name: str,
    experiment_name: str,
    task: Callable[..., Any],
    evaluators: list[Callable[..., Any]],
    metadata: Mapping[str, Any] | None = None,
    max_concurrency: int = 1,
):
    """Run Langfuse dataset experiment and return SDK result object."""
    client = _langfuse_client()
    dataset = client.get_dataset(dataset_name)
    result = dataset.run_experiment(
        name=experiment_name,
        task=task,
        evaluators=evaluators,
        max_concurrency=max_concurrency,
        metadata=dict(metadata or {}),
    )
    client.flush()
    return result


def shutdown() -> None:
    client = _langfuse_client()
    client.flush()
    client.shutdown()
