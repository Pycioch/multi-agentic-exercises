"""CLI entry point for step-7 evaluation workflow.

Commands:
- sync: ingest local qa_dataset.json into Opik dataset items
- run: execute in-process DeepEval evaluation
- all: sync + run
"""
from __future__ import annotations

import argparse
import os
import uuid

from cloudops.eval.dataset import build_opik_dataset_items, build_test_cases
from cloudops.eval.runner import run_question
from cloudops.observability.opik import log_eval_result, sync_dataset

DEFAULT_DATASET_NAME = "cloudops-step-07-eval"
ALLOWED_TIERS = {"l1", "l2"}


def _parse_tiers(raw: str) -> list[str]:
    tiers = [t.strip().lower() for t in raw.split(",") if t.strip()]
    invalid = [t for t in tiers if t not in ALLOWED_TIERS]
    if invalid:
        raise ValueError(f"Unsupported tier(s): {invalid}. Allowed: {sorted(ALLOWED_TIERS)}")
    return tiers


def _sync_dataset(dataset_name: str, tier: str | None) -> int:
    items = build_opik_dataset_items(tier=tier)
    inserted = sync_dataset(
        dataset_name=dataset_name,
        items=items,
        description="CloudOps step-07 deterministic eval dataset",
    )
    print(f"Opik dataset sync completed: {inserted}/{len(items)} items submitted.")
    return 0


def _run_tier_eval(*, dataset_name: str, tier: str, run_name: str, session_id: str) -> tuple[int, int]:
    """Run one tier in-process and return (total_cases, passed_cases)."""
    from cloudops.agents.react import build_agent
    from deepeval import evaluate
    from cloudops.eval.metrics import AnswerContainsMetric

    agent = build_agent()
    test_cases = build_test_cases(tier=tier)
    passed = 0

    for case in test_cases:
        actual = run_question(agent, question=case.input, run_name=run_name, session_id=session_id)
        case.actual_output = actual

        metric = AnswerContainsMetric(threshold=1.0)
        score = metric.measure(case)
        is_pass = metric.is_successful()
        if is_pass:
            passed += 1

        meta = dict(case.additional_metadata or {})
        log_eval_result(
            run_name=run_name,
            case_id=meta.get("invariant_id", "unknown"),
            dataset_name=dataset_name,
            question=case.input,
            actual_output=actual,
            metric_scores={metric.name: score},
            passed=is_pass,
            metadata=meta,
        )

    # Keep DeepEval programmatic path active (no subprocess) for parity with selected mode.
    evaluate(test_cases=test_cases, metrics=[AnswerContainsMetric(threshold=1.0)])
    return len(test_cases), passed


def _run_evaluation(dataset_name: str, tiers: list[str], run_prefix: str) -> int:
    os.environ["CLOUDOPS_OPIK_DATASET"] = dataset_name

    total_cases = 0
    total_passed = 0
    failed_tiers: list[str] = []

    for tier in tiers:
        run_name = f"{run_prefix}-{tier}"
        session_id = f"{run_prefix}-{tier}-{uuid.uuid4()}"
        os.environ["CLOUDOPS_EVAL_RUN_NAME"] = run_name
        os.environ["CLOUDOPS_EVAL_SESSION_ID"] = session_id
        print(f"Running in-process DeepEval for tier={tier} run={run_name}")
        try:
            tier_total, tier_passed = _run_tier_eval(
                dataset_name=dataset_name,
                tier=tier,
                run_name=run_name,
                session_id=session_id,
            )
        except ModuleNotFoundError as exc:
            print(f"Missing dependency while running evaluation: {exc}")
            return 2
        except Exception as exc:
            print(f"Tier {tier} failed with error: {exc}")
            failed_tiers.append(tier)
            continue

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
    parser = argparse.ArgumentParser(description="CloudOps step-07 evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sync_p = sub.add_parser("sync", help="Sync qa_dataset.json to Opik dataset")
    sync_p.add_argument("--dataset", default=os.getenv("CLOUDOPS_OPIK_DATASET", DEFAULT_DATASET_NAME))
    sync_p.add_argument("--tier", choices=["l1", "l2", "l3"], default=None)

    run_p = sub.add_parser("run", help="Run in-process DeepEval evaluation")
    run_p.add_argument("--dataset", default=os.getenv("CLOUDOPS_OPIK_DATASET", DEFAULT_DATASET_NAME))
    run_p.add_argument("--tiers", default="l1,l2", help="Comma-separated tiers: l1,l2")
    run_p.add_argument("--run-prefix", default="cloudops-step-07")

    all_p = sub.add_parser("all", help="Sync dataset then run in-process DeepEval evaluation")
    all_p.add_argument("--dataset", default=os.getenv("CLOUDOPS_OPIK_DATASET", DEFAULT_DATASET_NAME))
    all_p.add_argument("--tiers", default="l1,l2", help="Comma-separated tiers: l1,l2")
    all_p.add_argument("--run-prefix", default="cloudops-step-07")

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
