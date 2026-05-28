# cloudops-step-07 — Opik Tracing + DeepEval + Prompt Optimization

This step keeps runtime behavior from step 6 and replaces Langfuse observability with Opik.
It also adds prompt optimization using `opik-optimizer` and a DeepEval LLM-as-a-judge scoring bridge.

## What this step adds

| New/updated file | What it does |
|---|---|
| `src/cloudops/observability/opik.py` | Opik tracing config + dataset sync + per-case score logging |
| `src/cloudops/cli/eval.py` | Eval CLI (`sync`, `run`, `all`) now targeting Opik |
| `src/cloudops/cli/optimize.py` | Prompt optimization CLI (`cloudops-optimize`) |
| `src/cloudops/eval/dataset.py` | Adds `build_opik_dataset_items()` helper |
| `tests/eval/opik_logging.py` | Shared Opik logging helper for eval tests |
| `.env.example` | Opik keys + optimizer model variables |

All LLM calls are traced through Opik in both runtime and evaluation paths:
- `uv run cloudops`
- `uv run cloudops-eval run|all`

## Setup

```bash
cp .env.example .env
uv sync
uv run python scripts/generate_data.py   # if data/raw/ is empty
```

Set in `.env`:
- `OPENAI_API_KEY`
- `OPIK_API_KEY`
- `OPIK_WORKSPACE`
- optional `OPIK_PROJECT_NAME`
- optional `OPIK_URL_OVERRIDE`
- optional `CLOUDOPS_OPIK_DATASET`

## Run runtime agent

```bash
uv run cloudops
```

## Sync evaluation dataset to Opik

```bash
uv run cloudops-eval sync
```

Optional: sync only one tier

```bash
uv run cloudops-eval sync --tier l1
```

## Run in-process DeepEval (and mirror results to Opik)

```bash
uv run cloudops-eval run
```

Run one tier:

```bash
uv run cloudops-eval run --tiers l1
```

Sync and run in one command:

```bash
uv run cloudops-eval all
```

## Optimize prompt with Opik

```bash
uv run cloudops-optimize --tiers l1,l2 --samples 20
```

Useful knobs:
- `--meta-model` (optimizer model)
- `--task-model` (model used to answer dataset questions)
- `--judge-model` (DeepEval GEval judge model)
- `--dataset` (Opik dataset name)

## What to observe

1. `cloudops-eval sync` reports submitted item count to Opik dataset.
2. `cloudops-eval run` executes in-process DeepEval and logs scores (`AnswerContains`, `pass`) to Opik traces.
3. `cloudops` runtime traces are visible in Opik with per-request metadata and thread grouping.
4. `cloudops-optimize` records optimization trials and returns `best_score`.
