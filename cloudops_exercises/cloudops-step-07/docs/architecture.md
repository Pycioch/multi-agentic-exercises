# Step 7 — Architecture: Opik Tracing + DeepEval + Prompt Optimization

## Runtime graph (same as step 5)

```
START → planner → extractor → visualizer → END
```

Step 7 keeps runtime graph semantics and swaps observability from Langfuse to Opik.
Both runtime paths (`ReAct` and `Pipeline`) now pass Opik tracer callbacks,
so internal LLM invocations are captured in Opik traces.

## Eval + observability architecture

```
data/invariants.yaml
        │
        ▼
src/cloudops/eval/dataset.py
  ├── build_test_cases()              -> DeepEval LLMTestCase[]
  └── build_opik_dataset_items()      -> Opik dataset payloads
        │
        ├───────────────────────────────┐
        ▼                               ▼
cloudops-eval sync                 tests/eval/test_l1.py,test_l2.py
  (src/cloudops/cli/eval.py)         (in-process deepeval.evaluate)
        │                               │
        ▼                               ▼
Opik dataset + items              AnswerContainsMetric scoring
                                        │
                                        ▼
                          src/cloudops/observability/opik.py
                          trace + score logging in Opik
```

## Optimization path (new in step 7)

```
data/qa_dataset.json
        │
        ▼
cloudops-optimize
  (src/cloudops/cli/optimize.py)
        │
        ├── DeepEval GEval (LLM judge)
        ▼
MetaPromptOptimizer (opik-optimizer)
        ▼
Opik experiments + best prompt
```

## Eval tier contract (implemented in this step)

| Tier | Questions | Grading | Where visible |
|---|---|---|---|
| L1 | Single lookup facts | `AnswerContainsMetric` (0/1) | DeepEval output + Opik score |
| L2 | Aggregation facts | `AnswerContainsMetric` (0/1) | DeepEval output + Opik score |
| L3 | Open-ended reasoning | Prepared in dataset; not executed by default in CLI | Dataset items only |

## Key additions vs step 5

- `src/cloudops/observability/opik.py` — dataset ingestion and per-case score logging.
- `src/cloudops/cli/eval.py` — standalone eval CLI (`sync`, `run`, `all`) with Opik.
- `src/cloudops/cli/optimize.py` — prompt optimization command.
- `src/cloudops/eval/dataset.py` — Opik dataset-item export helper.
- `tests/eval/test_l1.py`, `tests/eval/test_l2.py` — DeepEval metrics mirrored to Opik.
- `.env.example` — Opik credentials and optimizer model variables.

## Independence guarantees

- All evaluation data stays local to step 6 (`data/invariants.yaml`, `data/qa_dataset.json`, generated `data/raw`).
- No code imports from other step directories.
- Full setup/sync/eval workflow is executable from this folder only.
