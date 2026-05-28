# Step 6 — Architecture: Independent Langfuse + DeepEval Eval Layer

## Runtime graph (same as step 5)

```
START → planner → extractor → visualizer → END
```

Step 6 does not change runtime graph semantics. It extends evaluation and observability.
Both runtime paths (`ReAct` and `Pipeline`) now pass Langfuse `CallbackHandler`
config so every internal LLM invocation is traced.

## Eval + observability architecture

```
data/invariants.yaml
        │
        ▼
src/cloudops/eval/dataset.py
  ├── build_test_cases()              -> DeepEval LLMTestCase[]
  └── build_langfuse_dataset_items()  -> Langfuse dataset payloads
        │
        ├───────────────────────────────┐
        ▼                               ▼
cloudops-eval sync                 tests/eval/test_l1.py,test_l2.py
  (src/cloudops/cli/eval.py)         (dataset.run_experiment)
        │                               │
        ▼                               ▼
Langfuse dataset + items          Experiment evaluators (AnswerContains + pass)
                                        │
                                        ▼
                          src/cloudops/observability/langfuse.py
                          experiment wrapper + score persistence in Langfuse
```

## Eval tier contract (implemented in this step)

| Tier | Questions | Grading | Where visible |
|---|---|---|---|
| L1 | Single lookup facts | `AnswerContainsMetric` (0/1) | DeepEval output + Langfuse score |
| L2 | Aggregation facts | `AnswerContainsMetric` (0/1) | DeepEval output + Langfuse score |
| L3 | Open-ended reasoning | Prepared in dataset; not executed by default in CLI | Dataset items only |

## Key additions vs step 5

- `src/cloudops/observability/langfuse.py` — dataset ingestion and per-case score logging.
- `src/cloudops/cli/eval.py` — standalone eval CLI (`sync`, `run`, `all`).
- `src/cloudops/eval/dataset.py` — Langfuse dataset-item export helper.
- `tests/eval/test_l1.py`, `tests/eval/test_l2.py` — DeepEval metrics are mirrored to Langfuse.
- `.env.example` — Langfuse credentials and dataset variables.

## Independence guarantees

- All evaluation data stays local to step 6 (`data/invariants.yaml`, `data/qa_dataset.json`, generated `data/raw`).
- No code imports from other step directories.
- Full setup/sync/eval workflow is executable from this folder only.
