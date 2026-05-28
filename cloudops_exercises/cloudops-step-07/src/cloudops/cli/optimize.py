"""CLI entry point for step-7 prompt optimization with Opik.

This command bridges DeepEval LLM-as-a-judge scoring into opik-optimizer.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from opik_optimizer import ChatPrompt, MetaPromptOptimizer

from cloudops.eval.dataset import load_qa_dataset
from cloudops.observability.opik import sync_dataset
from cloudops.prompts.system import SYSTEM_PROMPT

DEFAULT_DATASET_NAME = "cloudops-step-07-optimize"
DEFAULT_PROJECT_NAME = "cloudops-step-07-optimizer"


def _build_records(*, tiers: list[str]) -> list[dict[str, Any]]:
    rows = load_qa_dataset()
    return [
        {
            "item_id": row["id"],
            "question": row["question"],
            "expected_output": str(row["answer"]),
            "metadata": {
                "invariant_id": row["id"],
                "tier": row["tier"],
                "source_files": row.get("source_files", []),
            },
        }
        for row in rows
        if row.get("tier", "").lower() in tiers and row.get("answer") is not None
    ]


def _build_judge(model: str) -> GEval:
    return GEval(
        name="answer_correctness",
        criteria="Score whether actual output is correct and grounded in expected output.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.0,
        model=model,
        async_mode=False,
    )


def _build_metric(judge: GEval):
    def _metric(dataset_item: dict, llm_output: str) -> float:
        expected = dataset_item.get("expected_output")
        if not expected:
            return 0.0
        case = LLMTestCase(
            input=str(dataset_item.get("question", "")),
            actual_output=llm_output,
            expected_output=str(expected),
        )
        score = float(judge.measure(case))
        return max(0.0, min(1.0, score))

    return _metric


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudOps step-07 Opik optimizer CLI")
    parser.add_argument("--dataset", default=os.getenv("CLOUDOPS_OPIK_DATASET", DEFAULT_DATASET_NAME))
    parser.add_argument("--project", default=os.getenv("OPIK_PROJECT_NAME", DEFAULT_PROJECT_NAME))
    parser.add_argument("--tiers", default="l1,l2", help="Comma-separated tiers to optimize on.")
    parser.add_argument("--samples", type=int, default=20, help="Number of examples per trial.")
    parser.add_argument("--threads", type=int, default=8, help="Parallel threads for optimizer trials.")
    parser.add_argument("--meta-model", default=os.getenv("CLOUDOPS_OPIK_META_MODEL", "gpt-4o"))
    parser.add_argument("--judge-model", default=os.getenv("CLOUDOPS_OPIK_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--task-model", default=os.getenv("CLOUDOPS_OPIK_TASK_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    tiers = [t.strip().lower() for t in args.tiers.split(",") if t.strip()]
    records = _build_records(tiers=tiers)
    if not records:
        raise SystemExit(f"No optimization records found for tiers={tiers}.")

    os.environ["OPIK_PROJECT_NAME"] = args.project
    inserted = sync_dataset(
        dataset_name=args.dataset,
        items=records,
        description="CloudOps step-07 prompt optimization dataset",
    )
    print(f"Opik optimizer dataset prepared: {inserted}/{len(records)} items submitted.")

    from opik import Opik

    client = Opik(project_name=args.project)
    dataset = client.get_or_create_dataset(name=args.dataset)

    prompt = ChatPrompt(
        system=SYSTEM_PROMPT,
        user="{question}",
        model=args.task_model,
    )
    judge = _build_judge(model=args.judge_model)
    metric = _build_metric(judge)
    optimizer = MetaPromptOptimizer(model=args.meta_model, n_threads=args.threads)

    result = optimizer.optimize_prompt(
        prompt=prompt,
        dataset=dataset,
        metric=metric,
        n_samples=args.samples,
        project_name=args.project,
        experiment_config={
            "tiers": tiers,
            "step": "cloudops-step-07",
            "optimizer": "MetaPromptOptimizer",
            "judge_model": args.judge_model,
            "task_model": args.task_model,
        },
    )
    result.display()
    print(f"Best score: {result.score:.4f}")


if __name__ == "__main__":
    main()
