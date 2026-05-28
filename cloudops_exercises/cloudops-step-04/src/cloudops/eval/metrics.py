"""Custom DeepEval metrics for the CloudOps eval harness.

AnswerContainsMetric  — checks that a known ground-truth token appears verbatim
                        in the agent's actual output (L1/L2 use case ONLY).
                        Raises ValueError if called on an L3 case (no expected_output)
                        so the caller is forced to use an LLM-as-judge metric instead.

ToolCallVerifier      — checks that the agent called at least one expected tool
                        by scanning the output for '[tool: <name>]' markers injected
                        by the streaming CLI.  Graded 0/1 (binary).
"""
from __future__ import annotations

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class AnswerContainsMetric(BaseMetric):
    """Pass if expected_output appears (case-insensitive) in actual_output.

    Only valid for L1/L2 invariants that have a deterministic expected_output.
    Raises ValueError for L3 cases (expected_output=None) — use an LLM-as-judge
    metric for those instead.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.name = "AnswerContains"

    def measure(self, test_case: LLMTestCase) -> float:
        if test_case.expected_output is None:
            raise ValueError(
                "AnswerContainsMetric requires expected_output (L1/L2 only). "
                "L3 open-ended cases must be graded by an LLM-as-judge metric."
            )

        expected = str(test_case.expected_output).strip().lower()
        actual = str(test_case.actual_output).lower()
        self.score = 1.0 if expected in actual else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"Expected token '{expected}' {'found' if self.success else 'NOT found'} "
            f"in agent output."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class ToolCallVerifier(BaseMetric):
    """Pass if at least one of the required tools appears as a '[tool: <name>]' marker.

    The streaming CLI injects markers in the format '[tool: <name>]' for every
    tool call. This metric scans for those exact markers — not arbitrary substrings —
    to avoid false positives from the agent mentioning a tool name in prose.

    Pass required_tools=['query_csv'] to enforce that CSV lookup actually happened.
    """

    def __init__(
        self,
        required_tools: list[str],
        threshold: float = 1.0,
    ) -> None:
        self.threshold = threshold
        self.required_tools = required_tools
        self.name = "ToolCallVerifier"

    def measure(self, test_case: LLMTestCase) -> float:
        output = str(test_case.actual_output).lower()
        # match '[tool: <name>]' markers only — avoids prose false-positives
        called = [t for t in self.required_tools if f"[tool: {t.lower()}]" in output]
        self.score = 1.0 if called else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"Required tools {self.required_tools}; "
            f"markers found: {called if called else 'none'}."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success
