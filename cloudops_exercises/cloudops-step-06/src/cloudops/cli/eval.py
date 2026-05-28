"""CLI entry point for step-6 evaluation workflow.

Commands:
- sync: ingest local qa_dataset.json into Langfuse dataset items
- run: execute Langfuse dataset.run_experiment with deterministic evaluators
- all: sync + run
"""
from __future__ import annotations

import argparse
import os
import uuid

from cloudops.eval.dataset import build_langfuse_dataset_items
from cloudops.observability.langfuse import build_langchain_config, run_dataset_experiment, sync_dataset

DEFAULT_DATASET_NAME = "cloudops-step-06-eval"
ALLOWED_TIERS = {"l1", "l2"}


def _parse_tiers(raw: str) -> list[str]:
    tiers = [t.strip().lower() for t in raw.split(",") if t.strip()]
    invalid = [t for t in tiers if t not in ALLOWED_TIERS]
    if invalid:
        raise ValueError(f"Unsupported tier(s): {invalid}. Allowed: {sorted(ALLOWED_TIERS)}")
    return tiers


def _sync_dataset(dataset_name: str, tier: str | None) -> int:
    items = build_langfuse_dataset_items(tier=tier)
    inserted = sync_dataset(
        dataset_name=dataset_name,
        items=items,
        description="CloudOps step-06 deterministic eval dataset",
    )
    print(f"Langfuse dataset sync completed: {inserted}/{len(items)} items submitted.")
    return 0


def _run_question(agent, *, question: str, run_name: str, session_id: str) -> str:
    """Invoke the agent and return final text response."""
    config = build_langchain_config(
        run_name=run_name,
        session_id=session_id,
        tags=["cloudops-step-06", "eval", "deepeval"],
        metadata={"entrypoint": "cloudops-eval", "question_hash": str(hash(question))},
        configurable={"thread_id": f"eval-{hash(question)}"},
        fresh_handler=True,
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


def _run_tier_eval(*, dataset_name: str, tier: str, run_name: str, session_id: str) -> tuple[int, int]:
    """Run one tier in-process and return (total_cases, passed_cases)."""
    from cloudops.agents.react import build_agent
    from deepeval.test_case import LLMTestCase
    from cloudops.eval.metrics import AnswerContainsMetric
    from langfuse import Evaluation

    agent = build_agent()
    stats = {"total": 0, "passed": 0}

    def task(*, item, **kwargs):
        question = str(item.input["question"])
        stats["total"] += 1
        actual = _run_question(agent, question=question, run_name=run_name, session_id=session_id)
        return actual

    def answer_contains_evaluator(*, input, output, expected_output, item=None, **kwargs):
        if expected_output is None or "answer" not in expected_output:
            return Evaluation(name="AnswerContains", value=0.0)
        test_case = LLMTestCase(
            input=str(input["question"]),
            actual_output=str(output),
            expected_output=str(expected_output["answer"]),
        )
        metric = AnswerContainsMetric(threshold=1.0)
        score = metric.measure(test_case)
        return Evaluation(name="AnswerContains", value=float(score))

    def pass_evaluator(*, expected_output, output, item=None, **kwargs):
        if expected_output is None or "answer" not in expected_output:
            return Evaluation(name="pass", value=0.0)
        expected = str(expected_output["answer"]).strip().lower()
        is_pass = expected in str(output).lower()
        if is_pass:
            stats["passed"] += 1
        return Evaluation(name="pass", value=1.0 if is_pass else 0.0)

    run_dataset_experiment(
        dataset_name=dataset_name,
        experiment_name=run_name,
        task=task,
        evaluators=[answer_contains_evaluator, pass_evaluator],
        metadata={"tier": tier, "session_id": session_id, "mode": "hybrid-deepeval-bridge"},
        max_concurrency=1,
    )
    return stats["total"], stats["passed"]


def _run_evaluation(dataset_name: str, tiers: list[str], run_prefix: str) -> int:
    total_cases = 0
    total_passed = 0
    failed_tiers: list[str] = []

    for tier in tiers:
        tier_dataset_name = f"{dataset_name}-{tier}"
        os.environ["CLOUDOPS_LANGFUSE_DATASET"] = tier_dataset_name
        sync_code = _sync_dataset(dataset_name=tier_dataset_name, tier=tier)
        if sync_code != 0:
            return sync_code

        run_name = f"{run_prefix}-{tier}"
        session_id = f"{run_prefix}-{tier}-{uuid.uuid4()}"
        os.environ["CLOUDOPS_EVAL_RUN_NAME"] = run_name
        os.environ["CLOUDOPS_EVAL_SESSION_ID"] = session_id
        print(f"Running Langfuse experiment for tier={tier} run={run_name}")
        tier_total, tier_passed = _run_tier_eval(
            dataset_name=tier_dataset_name,
            tier=tier,
            run_name=run_name,
            session_id=session_id,
        )

        tier_failed = tier_total - tier_passed
        total_cases += tier_total
        total_passed += tier_passed
        print(f"Tier {tier}: passed={tier_passed} failed={tier_failed} total={tier_total}")

        if tier_failed > 0:
            failed_tiers.append(tier)

    total_failed = total_cases - total_passed
    print(
        f"Summary: tiers={tiers} passed={total_passed} failed={total_failed} total={total_cases}"
    )
    return 1 if failed_tiers else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudOps step-06 evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sync_p = sub.add_parser("sync", help="Sync qa_dataset.json to Langfuse dataset")
    sync_p.add_argument("--dataset", default=os.getenv("CLOUDOPS_LANGFUSE_DATASET", DEFAULT_DATASET_NAME))
    sync_p.add_argument("--tier", choices=["l1", "l2", "l3"], default=None)

    run_p = sub.add_parser("run", help="Run Langfuse experiment evaluation")
    run_p.add_argument("--dataset", default=os.getenv("CLOUDOPS_LANGFUSE_DATASET", DEFAULT_DATASET_NAME))
    run_p.add_argument("--tiers", default="l1,l2", help="Comma-separated tiers: l1,l2")
    run_p.add_argument("--run-prefix", default="cloudops-step-06")

    all_p = sub.add_parser("all", help="Sync dataset then run Langfuse experiment evaluation")
    all_p.add_argument("--dataset", default=os.getenv("CLOUDOPS_LANGFUSE_DATASET", DEFAULT_DATASET_NAME))
    all_p.add_argument("--tiers", default="l1,l2", help="Comma-separated tiers: l1,l2")
    all_p.add_argument("--run-prefix", default="cloudops-step-06")

    args = parser.parse_args()

    if args.command == "sync":
        raise SystemExit(_sync_dataset(dataset_name=args.dataset, tier=args.tier))

    tiers = _parse_tiers(args.tiers)

    if args.command == "run":
        raise SystemExit(_run_evaluation(dataset_name=args.dataset, tiers=tiers, run_prefix=args.run_prefix))

    # args.command == "all"
    sync_code = _sync_dataset(dataset_name=args.dataset, tier=None)
    if sync_code != 0:
        raise SystemExit(sync_code)
    raise SystemExit(_run_evaluation(dataset_name=args.dataset, tiers=tiers, run_prefix=args.run_prefix))


if __name__ == "__main__":
    main()
