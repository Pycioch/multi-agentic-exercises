# cloudops-step-06 — Langfuse Experiments + Deterministic Eval Bridge

**What this step adds** (vs step-05):

| New file | What it does |
|---|---|
| `src/cloudops/observability/langfuse.py` | Langfuse dataset sync + `dataset.run_experiment(...)` wrapper |
| `src/cloudops/cli/eval.py` | CLI wrapper to sync dataset and run experiment-first eval |
| `src/cloudops/observability/__init__.py` | Observability package marker |
| `src/cloudops/eval/dataset.py` | Adds `build_langfuse_dataset_items()` export helper |
| `tests/eval/test_l1.py`, `test_l2.py` | Logs DeepEval metric scores and pass/fail to Langfuse |
| `.env.example` | Adds Langfuse keys and dataset name |

Runtime path from step 5 is preserved (`cloudops` CLI, pipeline routing, budget/input rails), while step 6 eval uses Langfuse experiments as the canonical run path.

All LLM calls are traced through Langfuse in both modes:
- `uv run cloudops` (runtime ReAct + pipeline calls)
- `uv run cloudops-eval run|all` (Langfuse experiment task runs)

## Eval data flow

```
data/qa_dataset.json
        ↓
cloudops-eval sync
        ↓
Langfuse dataset + items
        ↓
cloudops-eval run
        ↓
dataset.run_experiment(task=agent_call, evaluators=[...])
        ↓
Langfuse experiment scores (AnswerContains + pass)
```

## Setup

```bash
cp .env.example .env
uv sync
uv run python scripts/generate_data.py   # if data/raw/ is empty
```

Set in `.env`:
- `OPENAI_API_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- optional `LANGFUSE_HOST`
- optional `CLOUDOPS_LANGFUSE_DATASET`

## Run the agent

```bash
uv run cloudops
```

## Sync evaluation dataset to Langfuse

```bash
uv run cloudops-eval sync
```

Optional: sync only one tier

```bash
uv run cloudops-eval sync --tier l1
```

## Run Langfuse experiment eval from CLI

```bash
uv run cloudops-eval run
```

This command runs Langfuse `dataset.run_experiment(...)` with deterministic
DeepEval-style evaluator semantics.

Run one tier:

```bash
uv run cloudops-eval run --tiers l1
```

Sync and run in one command:

```bash
uv run cloudops-eval all
```

## What to observe

1. `cloudops-eval sync` reports how many items were submitted to the Langfuse dataset.
2. `cloudops-eval run` executes Langfuse experiment runs for selected tiers.
3. Every LLM call from `cloudops` and `cloudops-eval` appears in Langfuse traces.
4. Experiment evaluators write Langfuse scores:
   - `AnswerContains` (0/1)
   - `pass` (0/1)
5. Test metadata includes invariant id and tier to filter traces per dataset item.

## What changed vs step-05

| File | Change |
|---|---|
| `src/cloudops/observability/langfuse.py` | New |
| `src/cloudops/observability/__init__.py` | New |
| `src/cloudops/cli/eval.py` | New experiment-first eval runner |
| `src/cloudops/eval/dataset.py` | Added Langfuse dataset item export helper |
| `tests/eval/test_l1.py` | Added Langfuse metric logging |
| `tests/eval/test_l2.py` | Added Langfuse metric logging |
| `.env.example` | Added Langfuse variables |
| `pyproject.toml` | Step name/description + `langfuse` dependency + `cloudops-eval` script |
