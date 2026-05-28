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
from cloudops.agents.react import build_agent

# ── shared agent (one per test session) ──────────────────────────────────────

@pytest.fixture(scope="module")
def agent():
    return build_agent()


def _run_question(agent, question: str) -> str:
    """Invoke the agent and return the final AI text reply."""
    config = {"configurable": {"thread_id": f"eval-{hash(question)}"}}
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


# ── parametrised test cases ───────────────────────────────────────────────────

_L1_CASES = build_test_cases(tier="l1")


@pytest.mark.parametrize(
    "test_case",
    _L1_CASES,
    ids=[tc.additional_metadata["invariant_id"] for tc in _L1_CASES],
)
def test_l1_invariant(agent, test_case: LLMTestCase):
    actual = _run_question(agent, test_case.input)
    test_case.actual_output = actual

    metric = AnswerContainsMetric(threshold=1.0)
    metric.measure(test_case)

    assert_test(
        test_case,
        metrics=[metric],
    )
