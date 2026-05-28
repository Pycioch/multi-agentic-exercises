# Step 2 — Architecture: ReAct Agent + Analytics Tools

## Graph (conceptual — no LangGraph yet)

```
User CLI input
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                ReAct Agent (GPT-4o)                  │
│                                                      │
│  Tools available:                                    │
│  ├── list_csv_files()          ← step 1              │
│  ├── describe_csv(filename)    ← step 1              │
│  ├── query_csv(filename, …)   ← step 1              │
│  ├── aggregate_csv(…)         ← NEW step 2          │
│  ├── timeseries_resample(…)   ← NEW step 2          │
│  ├── compute_correlation(…)   ← NEW step 2          │
│  ├── top_n(…)                 ← NEW step 2          │
│  ├── plot_timeseries(…)       ← NEW step 2          │
│  ├── plot_bar(…)              ← NEW step 2          │
│  └── plot_histogram(…)        ← NEW step 2          │
└─────────────────────────────────────────────────────┘
      │                    │
      ▼                    ▼
   Text answer          PNG chart
                    (data/raw/charts/)
```

## Key additions vs step 1

- `pandas_tools.py` — aggregation, resampling, correlation, top-N using pandas DataFrames.
- `viz_tools.py` — Matplotlib charts saved to disk (`matplotlib.use("Agg")` for headless rendering).
- All new tools support `date_from`/`date_to` range filters and `filter_column`/`filter_value` equality filters.
- `_apply_filters()` helper shared across all pandas tools.

## Module tie: Workshop M3 — "Single-agent ReAct and when it is enough"
