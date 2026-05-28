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
from cloudops.agents.react import build_agent


@pytest.fixture(scope="module")
def agent():
    return build_agent()


def _run_question(agent, question: str) -> str:
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


_L2_CASES = build_test_cases(tier="l2")


@pytest.mark.parametrize(
    "test_case",
    _L2_CASES,
    ids=[tc.additional_metadata["invariant_id"] for tc in _L2_CASES],
)
def test_l2_invariant(agent, test_case: LLMTestCase):
    actual = _run_question(agent, test_case.input)
    test_case.actual_output = actual

    assert_test(
        test_case,
        metrics=[AnswerContainsMetric(threshold=1.0)],
    )
