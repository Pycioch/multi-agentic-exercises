# cloudops-step-04 — Eval harness bootstrap

**What this step adds** (vs step-03):
- `eval/invariants.py` — typed `Invariant` dataclass + `load()` / `by_id()` helpers
- `eval/dataset.py` — derives `LLMTestCase` objects from invariants; persists `data/qa_dataset.json`
- `eval/metrics.py` — `AnswerContainsMetric` (verbatim token check) + `ToolCallVerifier`
- `tests/eval/test_l1.py` — DeepEval pytest gate; runs all 10 L1 invariants as CI assertions

**Learning goal (M7):** deterministic L1 eval is the floor. L1 uses ground-truth tokens that
must appear verbatim in the agent output — no LLM judge needed. L3 cases intentionally have
no `expected_output` to show where automated eval ends and human/LLM-judge judgment begins.

## Setup

```bash
cp .env.example .env
uv sync
uv run python scripts/generate_data.py   # generates the 7 CSVs
uv run python -m cloudops.eval.dataset   # derives data/qa_dataset.json
```

## Run the agent

```bash
uv run cloudops
```

## Unit tests

```bash
uv run pytest tests/unit/
```

## Eval tests (DeepEval)

```bash
uv run deepeval test run tests/eval/test_l1.py
```

> Use `deepeval test run`, not plain `pytest` — DeepEval needs its own runner to record
> results and generate the structured report.

## What changed

| File | Change |
|---|---|
| `eval/invariants.py` | New — typed loader for `data/invariants.yaml` |
| `eval/dataset.py` | New — derives `LLMTestCase` list; saves `data/qa_dataset.json` |
| `eval/metrics.py` | New — `AnswerContainsMetric`, `ToolCallVerifier` |
| `tests/eval/test_l1.py` | New — parametrised DeepEval gate for all L1 invariants |
| `pyproject.toml` | name updated; `deepeval>=4.0.2` added |

## Eval tiers

| Tier | Count | Ground truth | Graded by |
|------|-------|--------------|-----------|
| L1 | 10 | Exact token from invariants.yaml | `AnswerContainsMetric` (pytest CI) |
| L2 | 10 | Exact token from invariants.yaml | `AnswerContainsMetric` (pytest CI) |
| L3 | 10 | None | LLM-as-judge only (step 7+) |

## What to observe

1. **Run the eval gate** — all 10 L1 tests should pass against the synthetic CSVs.

2. **Break one invariant** — edit `data/qa_dataset.json` to change an L1 answer,
   re-run `deepeval test run tests/eval/test_l1.py` and watch it fail. This is the
   regression guard that runs on every PR from step 4 onward.

3. **Inspect L3 cases** — call `build_test_cases(tier="l3")` and note that
   `expected_output` is `None`. These are graded only by LLM-as-judge in step 7.
