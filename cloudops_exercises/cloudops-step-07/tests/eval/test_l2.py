"""DeepEval CI gate for L2 (aggregation) invariants.

L2 questions require the agent to aggregate data across multiple rows
(counts, means, joins) — not just look up a single value.

Run with:  deepeval test run tests/eval/test_l2.py
"""
from __future__ import annotations

import pytest

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from cloudops.eval.dataset import build_test_cases
from cloudops.eval.metrics import AnswerContainsMetric
from cloudops.eval.runner import run_question
from cloudops.agents.react import build_agent
from tests.eval.opik_logging import log_test_case_result


@pytest.fixture(scope="module")
def agent():
    return build_agent()


_L2_CASES = build_test_cases(tier="l2")


@pytest.mark.parametrize(
    "test_case",
    _L2_CASES,
    ids=[tc.additional_metadata["invariant_id"] for tc in _L2_CASES],
)
def test_l2_invariant(agent, test_case: LLMTestCase):
    actual = run_question(
        agent,
        question=test_case.input,
        run_name="cloudops-step-07-l2",
        session_id="cloudops-step-07-eval-tests",
    )
    test_case.actual_output = actual

    metric = AnswerContainsMetric(threshold=1.0)
    score = metric.measure(test_case)
    passed = metric.is_successful()

    log_test_case_result(
        default_run_name="cloudops-step-07-l2",
        test_case=test_case,
        actual_output=actual,
        metric_name=metric.name,
        metric_score=score,
        passed=passed,
    )

    assert_test(
        test_case,
        metrics=[metric],
    )
