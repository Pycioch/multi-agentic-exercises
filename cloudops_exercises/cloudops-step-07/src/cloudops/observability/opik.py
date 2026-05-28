"""Opik helpers for tracing, dataset sync, and eval score logging.

The module is intentionally tolerant: if Opik credentials/config are missing,
all operations become no-ops so local development still works.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

_CLIENT = None


def _project_name(default: str = "cloudops-step-07") -> str:
    return os.getenv("OPIK_PROJECT_NAME", default).strip() or default


def _opik_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    has_api_key = bool(os.getenv("OPIK_API_KEY", "").strip())
    use_local = os.getenv("OPIK_USE_LOCAL", "").strip().lower() in {"1", "true", "yes"}
    if not has_api_key and not use_local:
        return None

    try:
        import opik
    except Exception:
        return None

    workspace = os.getenv("OPIK_WORKSPACE", "").strip() or None
    host = os.getenv("OPIK_URL_OVERRIDE", "").strip() or None
    try:
        _CLIENT = opik.Opik(project_name=_project_name(), workspace=workspace, host=host)
        return _CLIENT
    except Exception:
        return None


def opik_enabled() -> bool:
    return _opik_client() is not None


def build_langchain_config(
    *,
    run_name: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    configurable: Mapping[str, Any] | None = None,
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    """Build RunnableConfig with Opik tracer callback when available."""
    cfg: dict[str, Any] = {}
    if configurable:
        cfg["configurable"] = dict(configurable)
    if recursion_limit is not None:
        cfg["recursion_limit"] = recursion_limit

    md = dict(metadata or {})
    md["opik_trace_name"] = run_name
    if session_id:
        md["opik_thread_id"] = session_id
    if user_id:
        md["opik_user_id"] = user_id
    if tags:
        md["opik_tags"] = tags
    if md:
        cfg["metadata"] = md

    try:
        from opik.integrations.langchain import OpikTracer

        tracer = OpikTracer(
            project_name=_project_name(),
            thread_id=session_id,
            tags=tags or [],
            metadata=md or None,
        )
        cfg["callbacks"] = [tracer]
    except Exception:
        # Tracing is optional; runtime should proceed without failures.
        pass

    return cfg


def sync_dataset(dataset_name: str, items: list[dict], description: str = "") -> int:
    """Create/update Opik dataset and insert items.

    Returns number of inserted items. If Opik is unavailable, returns 0.
    """
    client = _opik_client()
    if client is None:
        return 0

    try:
        dataset = client.get_or_create_dataset(name=dataset_name)
    except Exception:
        return 0

    payloads: list[dict] = []
    for item in items:
        payloads.append(
            {
                "item_id": item.get("item_id"),
                "question": item.get("question") or item.get("input"),
                "expected_output": item.get("expected_output"),
                "metadata": item.get("metadata", {}),
            }
        )
    try:
        dataset.insert(payloads)
        return len(payloads)
    except Exception:
        return 0


def log_eval_result(
    *,
    run_name: str,
    case_id: str,
    dataset_name: str,
    question: str,
    actual_output: str,
    metric_scores: Mapping[str, float],
    passed: bool,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Log one evaluation case as Opik trace + feedback scores."""
    client = _opik_client()
    if client is None:
        return

    trace_metadata = dict(metadata or {})
    trace_metadata.update({"case_id": case_id, "dataset_name": dataset_name})

    try:
        trace = client.trace(
            name=run_name,
            input={"question": question},
            output={"answer": actual_output},
            metadata=trace_metadata,
            tags=["cloudops-step-07", "eval", "deepeval"],
            thread_id=os.getenv("CLOUDOPS_EVAL_SESSION_ID", "cloudops-step-07-eval"),
        )
        scores = [
            {"id": trace.id, "name": metric_name, "value": float(value), "reason": f"DeepEval metric {metric_name}"}
            for metric_name, value in metric_scores.items()
        ]
        scores.append(
            {
                "id": trace.id,
                "name": "pass",
                "value": 1.0 if passed else 0.0,
                "reason": "Binary pass/fail for this evaluation case",
            }
        )
        client.log_traces_feedback_scores(scores=scores)
        client.flush()
    except Exception:
        return


def shutdown() -> None:
    client = _opik_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass
    try:
        client.end()
    except Exception:
        pass
