# Step 4 — Architecture: ReAct Agent + Eval Harness

## Graph (unchanged from step 3)

Same ReAct + HITL graph as step 3. No new agent capabilities added in step 4.

## Eval harness architecture

```
data/invariants.yaml   (30 ground-truth facts: 10 L1, 10 L2, 10 L3)
        │
        ▼
scripts/generate_data.py
        │  (seeded, reproducible; plants anomalies from invariants)
        ▼
data/raw/
  ├── incidents.csv        (~520K rows)
  ├── metrics_cpu.csv      (~1.6M rows)
  ├── metrics_mem.csv      (~1.6M rows)
  ├── metrics_net.csv      (~1.6M rows)
  ├── deployments.csv      (~3K rows)
  ├── oncall_roster.csv    (~6K rows)
  └── runbooks.csv         (~100 rows)
        │
        ▼
src/cloudops/eval/
  ├── invariants.py     — loads invariants.yaml → Invariant dataclass
  ├── dataset.py        — invariant → LLMTestCase; saves qa_dataset.json
  └── metrics.py        — AnswerContainsMetric (L1/L2) + ToolCallVerifier (L1/L2)
        │
        ▼
tests/eval/
  ├── test_l1.py   — DeepEval pytest: exact lookup questions
  └── test_l2.py   — DeepEval pytest: aggregation questions
```

## Eval tier contract

| Tier | Questions | Grading | Metric |
|---|---|---|---|
| L1 | Single lookup (who, what, which) | `AnswerContainsMetric` (substring) | pass/fail |
| L2 | Aggregation (count, mean, sum) | `AnswerContainsMetric` (substring) | pass/fail |
| L3 | Causal/temporal reasoning | LLM-as-judge only (no `expected_output`) | qualitative |

## Key additions vs step 3

- `eval/invariants.py` — typed loader for `data/invariants.yaml`.
- `eval/dataset.py` — derives `LLMTestCase` per invariant; saves `data/qa_dataset.json`.
- `eval/metrics.py` — `AnswerContainsMetric` (requires `expected_output`; raises on L3) + `ToolCallVerifier`.
- `tests/eval/test_l1.py`, `test_l2.py` — `@pytest.mark.parametrize` over invariant IDs.
- `scripts/generate_data.py` — plants 4 specific events (EVT-A through EVT-E) traceable to invariants.
- `data/invariants.yaml` — 30 invariants (≥30 per plan requirement).
- `data/qa_dataset.json` — derived dataset (30 Q&A pairs).

## Module tie: Workshop M7 — "Observability + evaluation + optimization"
