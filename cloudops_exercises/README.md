# CloudOps Exercises

Progressive CloudOps agent engineering track built as 7 incremental steps.
Each step is self-contained and has its own `README.md`, dependencies, data generator, and tests.

## Track overview

| Step | Focus | What changes |
|---|---|---|
| `cloudops-step-01` | Single ReAct baseline | CSV browsing tools + interactive CLI |
| `cloudops-step-02` | Better tools first | pandas analytics + chart tools, same architecture |
| `cloudops-step-03` | Human-in-the-loop | `interrupt()` + resume with `Command(resume=...)` |
| `cloudops-step-04` | Deterministic evaluation | DeepEval harness + invariant-driven dataset |
| `cloudops-step-05` | Graph architecture | Planner/Extractor/Visualizer pipeline + routing + budget + input rails |
| `cloudops-step-06` | Observability and experiments | Langfuse tracing + experiment-first eval CLI |
| `cloudops-step-07` | Production eval loop | Opik tracing + DeepEval bridge + prompt optimization |

## How to use this directory

- Start from `cloudops-step-01`.
- Move step-by-step to keep the learning progression and code diffs understandable.
- Use each step's local `README.md` for exact setup/run/test commands.
- Treat each step as standalone; do not assume a shared virtual environment.

## Quick path

```bash
cd cloudops_exercises/cloudops-step-01
cp .env.example .env
uv sync
uv run python scripts/generate_data.py
uv run cloudops
```

Then continue with:

- `cloudops-step-02` to extend capabilities via tools.
- `cloudops-step-03` to add HITL.
- `cloudops-step-04` to bootstrap eval.
- `cloudops-step-05` to introduce pipeline routing and controls.
- `cloudops-step-06` and `cloudops-step-07` for observability and optimization workflows.

## Notes

- Runtime and eval commands evolve per step; always use the command forms documented in that step.
- API keys and observability variables differ between later steps (Langfuse in step 06, Opik in step 07).
