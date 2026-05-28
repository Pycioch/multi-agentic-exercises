"""Shared Opik score logging helper for eval tests."""
from __future__ import annotations

import os

from deepeval.test_case import LLMTestCase

from cloudops.observability.opik import log_eval_result


def log_test_case_result(
    *,
    default_run_name: str,
    test_case: LLMTestCase,
    actual_output: str,
    metric_name: str,
    metric_score: float,
    passed: bool,
) -> None:
    meta = dict(test_case.additional_metadata or {})
    dataset_name = os.getenv("CLOUDOPS_OPIK_DATASET", "cloudops-step-07-eval")
    run_name = os.getenv("CLOUDOPS_EVAL_RUN_NAME", default_run_name)
    log_eval_result(
        run_name=run_name,
        case_id=meta.get("invariant_id", "unknown"),
        dataset_name=dataset_name,
        question=test_case.input,
        actual_output=actual_output,
        metric_scores={metric_name: metric_score},
        passed=passed,
        metadata=meta,
    )
