"""Derive Q&A test cases from invariants.yaml.

Each invariant produces one LLMTestCase with a deterministic expected answer.
~20% of cases use only LLM-as-judge (no expected_output) to teach the limits
of automated eval — these are the L3 open-ended entries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

try:
    from deepeval.test_case import LLMTestCase
except Exception:
    @dataclass
    class LLMTestCase:
        """Fallback model used only when deepeval is not installed locally."""

        input: str
        actual_output: str
        expected_output: str | None = None
        additional_metadata: dict | None = None

from cloudops.eval.invariants import Invariant, load

_ROOT = Path(__file__).parents[3]
_QA_PATH = _ROOT / "data" / "qa_dataset.json"


def _invariant_to_case(inv: Invariant) -> LLMTestCase:
    """Convert one invariant to a DeepEval test case.

    L1/L2 invariants get expected_output (deterministic ground truth).
    L3 invariants are left without expected_output — graded by LLM-as-judge only.
    """
    expected = str(inv.answer) if inv.tier in ("l1", "l2") else None
    return LLMTestCase(
        input=inv.question,
        actual_output="",   # filled at eval time by the agent under test
        expected_output=expected,
        additional_metadata={"invariant_id": inv.id, "tier": inv.tier},
    )


def build_test_cases(tier: str | None = None) -> list[LLMTestCase]:
    """Return test cases for the given tier ("l1"/"l2"/"l3") or all tiers."""
    all_inv = load()
    tiers = [tier] if tier else ["l1", "l2", "l3"]
    cases = []
    for t in tiers:
        for inv in all_inv[t]:
            cases.append(_invariant_to_case(inv))
    return cases


def save_qa_dataset(path: Path = _QA_PATH) -> None:
    """Persist the derived Q&A dataset as JSON (checked into the repo)."""
    all_inv = load()
    records = []
    for tier_invs in all_inv.values():
        for inv in tier_invs:
            records.append({
                "id": inv.id,
                "tier": inv.tier,
                "question": inv.question,
                "answer": inv.answer,
                "source_files": inv.source_files,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Saved {len(records)} Q&A pairs → {path}")


def load_qa_dataset(path: Path = _QA_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m cloudops.eval.dataset"
        )
    return json.loads(path.read_text())


def build_opik_dataset_items(
    path: Path = _QA_PATH,
    tier: str | None = None,
) -> list[dict]:
    """Return deterministic dataset-item payloads for Opik ingestion.

    Each item contains:
    - item_id: stable invariant id
    - input: question
    - expected_output: deterministic answer (L1/L2) or None (L3)
    - metadata: tier + source files
    """
    rows = load_qa_dataset(path)
    if tier:
        rows = [row for row in rows if row.get("tier") == tier]
    items: list[dict] = []
    for row in rows:
        item_tier = str(row.get("tier", "")).lower()
        answer = row.get("answer")
        expected = str(answer) if item_tier in ("l1", "l2") and answer is not None else None
        items.append(
            {
                "item_id": row["id"],
                "question": row["question"],
                "input": row["question"],
                "expected_output": expected,
                "metadata": {
                    "invariant_id": row["id"],
                    "tier": item_tier,
                    "source_files": row.get("source_files", []),
                },
            }
        )
    return items


if __name__ == "__main__":
    save_qa_dataset()
