# cloudops-step-01 — ReAct agent + CSV-browsing tools

**What this step adds** (baseline — no previous step):
- `agents/react.py` — single ReAct agent built with LangChain 1.3.1 `create_agent()`
- `tools/csv_tools.py` — three tools: `list_csv_files`, `describe_csv`, `query_csv`
- `prompts/system.py` — system prompt defining the analyst persona
- `cli/run.py` — interactive REPL with streaming tool-call display

**Learning goal (M3):** a single ReAct agent is the right answer for Level-1 questions
(single datacenter, single metric, point-in-time lookup). Multi-agent is not needed here.

## Setup

```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY
uv sync
uv run python scripts/generate_data.py   # create synthetic CSVs (step 4 adds this)
```

## Run

```bash
uv run cloudops
# or
uv run python -m cloudops.cli.run
```

## Test

```bash
uv run pytest tests/unit/
```

## Key files

| File | Purpose |
|---|---|
| `src/cloudops/core/config.py` | `pydantic-settings` config; reads `.env` |
| `src/cloudops/tools/csv_tools.py` | `@tool`-decorated pandas CSV readers |
| `src/cloudops/agents/react.py` | `create_agent()` wiring LLM + tools |
| `src/cloudops/cli/run.py` | REPL; streams tool calls to console |

## What to observe

1. Ask a Level-1 question: `"How many P0 incidents occurred in DC-A?"`
   → agent calls `describe_csv` then `query_csv` once. Single-agent is sufficient.

2. Ask a Level-3 question: `"Did the 2025-03-15 deploy correlate with the DC-A CPU spike?"`
   → agent makes many sequential tool calls across multiple files. It works, but watch
   the latency and token cost — this motivates the Orchestrator-Worker pattern in step 9.
