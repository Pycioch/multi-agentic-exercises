# Step 1 — Architecture: Single ReAct Agent

## Graph (conceptual — no LangGraph yet)

```
User CLI input
      │
      ▼
┌─────────────────────────────────────────┐
│          ReAct Agent (GPT-4o)            │
│                                          │
│  Thought → Action → Observation loop    │
│                                          │
│  Tools available:                        │
│  ├── list_csv_files()                   │
│  ├── describe_csv(filename)             │
│  └── query_csv(filename, filters…)     │
└─────────────────────────────────────────┘
      │
      ▼
   Answer (text)
```

## Data flow

```
CLI (run.py)
  → agent.stream({"messages": [("user", query)]})
  → ChatOpenAI (GPT-4o)
  → @tool calls routed by LangChain
  → CSV reads via pandas (data/raw/*.csv)
  → AIMessage with final answer
  → Console.print(Markdown(answer))
```

## Key design decisions

- `create_agent()` from LangChain 1.3.1 — wraps LangGraph's prebuilt ReAct loop.
- `MemorySaver` checkpointer — in-process, one `thread_id` per CLI session.
- No HITL, no analytics tools, no charts — single data-browsing capability only.
- `data_dir` configured via pydantic-settings (`.env` or env var).

## Module tie: Workshop M3 — "Single-agent ReAct and when it is enough"
