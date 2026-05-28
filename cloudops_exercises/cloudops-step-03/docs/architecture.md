# Step 3 — Architecture: ReAct Agent + HITL

## Graph (LangGraph with interrupt)

```
User CLI input
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                ReAct Agent (GPT-4o)                  │
│                                                      │
│  Tools available:                                    │
│  ├── list_csv_files / describe_csv / query_csv      │
│  ├── aggregate_csv / timeseries_resample / top_n    │
│  ├── compute_correlation                            │
│  ├── plot_timeseries / plot_bar / plot_histogram    │
│  └── ask_human(question)  ← NEW step 3             │
│          │                                           │
│          └──► interrupt() suspends graph             │
└─────────────────────────────────────────────────────┘
      │                              │
      │                    GraphInterrupt raised
      ▼                              │
   Answer (text)           CLI catches → prompts user
                                     │
                              Command(resume=answer)
                                     │
                              graph.stream() resumed
```

## HITL mechanics

1. Agent calls `ask_human(question)` when genuinely blocked.
2. `interrupt({"question": question})` suspends the LangGraph execution.
3. `GraphInterrupt` propagates to `cli/run.py` which catches it by class name.
4. CLI prints the question, collects `input()`, calls `_run_turn(resume_value=answer)`.
5. `Command(resume=answer)` resumes the graph from the suspension point.
6. `MemorySaver` preserves full graph state across the interrupt.

## Key additions vs step 2

- `ask_human` tool with `interrupt()` — Human-In-The-Loop pattern.
- `core/state.py` — typed state (TypedDict) introduced here, pre-LangGraph explicit graph.
- `run.py` HITL loop handles `GraphInterrupt` with recursive `_run_turn`.
- `isinstance(update, dict)` guard in stream loop (LangGraph emits tuples in some resume chunks).

## Known limitation (teaching point)

GPT-4o may ignore prompt-level HITL instructions probabilistically. This is intentional — step 3 demonstrates the "probabilistic wall" from `theory_research/when_to_use_what/strategy_regulated_industries.md`. Step 5 solves it via graph-level enforcement.

## Module tie: Workshop M3 — "Single-agent ReAct and when it is enough"
