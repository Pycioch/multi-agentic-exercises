"""DeepEval CI gate for L1 (deterministic lookup) invariants.

Run with:  deepeval test run tests/eval/test_l1.py
NOT with plain pytest — DeepEval's assert_test must be used to record results
in its internal dashboard and produce the structured report.

Each test:
1. Sends the invariant question to the built agent.
2. Captures the final text reply.
3. Asserts the expected answer token is present (AnswerContainsMetric).

The agent is built once per session via a module-level fixture.
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

# ── shared agent (one per test session) ──────────────────────────────────────

@pytest.fixture(scope="module")
def agent():
    return build_agent()


# ── parametrised test cases ───────────────────────────────────────────────────

_L1_CASES = build_test_cases(tier="l1")


@pytest.mark.parametrize(
    "test_case",
    _L1_CASES,
    ids=[tc.additional_metadata["invariant_id"] for tc in _L1_CASES],
)
def test_l1_invariant(agent, test_case: LLMTestCase):
    actual = run_question(
        agent,
        question=test_case.input,
        run_name="cloudops-step-07-l1",
        session_id="cloudops-step-07-eval-tests",
    )
    test_case.actual_output = actual

    metric = AnswerContainsMetric(threshold=1.0)
    score = metric.measure(test_case)
    passed = metric.is_successful()

    log_test_case_result(
        default_run_name="cloudops-step-07-l1",
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
