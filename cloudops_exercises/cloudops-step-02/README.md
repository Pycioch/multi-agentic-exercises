# cloudops-step-02 — pandas analytics tools + chart rendering

**What this step adds** (vs step-01):
- `tools/pandas_tools.py` — 4 analytics tools: `aggregate_csv`, `timeseries_resample`, `compute_correlation`, `top_n`
- `tools/viz_tools.py` — 3 chart tools: `plot_timeseries`, `plot_bar`, `plot_histogram`
- `core/config.py` — adds `charts_dir` (default: `data/charts`)

**Learning goal (M3):** the same single ReAct agent now answers aggregation and correlation
questions that step-01 could not — without touching the agent architecture.
Tool design, not agent topology, is the first lever to reach for.

## Setup

```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY
uv sync
uv run python scripts/generate_data.py   # if not done in step-01
```

## Run

```bash
uv run cloudops
```

## Test

```bash
uv run pytest tests/unit/
```

## New tools

| Tool | Purpose |
|---|---|
| `aggregate_csv` | Group-by + count/mean/sum/min/max/std |
| `timeseries_resample` | Resample timeseries to coarser frequency (e.g. 1h, 1D) |
| `compute_correlation` | Pearson r between two columns, optionally across files |
| `top_n` | Sort by column, return top N rows |
| `plot_timeseries` | Line chart → saved PNG in `data/charts/` |
| `plot_bar` | Bar chart with group-by aggregation |
| `plot_histogram` | Distribution of a numeric column |

## What to observe

1. **L2 question:** `"What is the mean MTTR for P0 incidents in DC-A?"`
   → agent calls `aggregate_csv` once. Compare with step-01 where this required manual filtering.

2. **L3 correlation:** `"Does the CPU spike in DC-A correlate with the 2025-03-15 P0 cluster?"`
   → agent chains `timeseries_resample` then `compute_correlation`. Count the tool calls —
   this sequential cost motivates the Orchestrator-Worker pattern in step-09.

3. **Chart:** `"Plot CPU on dc-a-host-02 around the 2025-03-15 outage"`
   → agent calls `plot_timeseries` and returns a file path you can open locally.
