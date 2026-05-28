# cloudops-step-03 — HITL via `interrupt()`

**What this step adds** (vs step-02):
- `core/state.py` — `AgentState` TypedDict (preview of the state model used from step-05)
- `agents/react.py` — `ask_human` tool using `interrupt()`; agent compiled with `MemorySaver`
- `cli/run.py` — `_run_turn()` catches `GraphInterrupt`, prompts the user, resumes via `Command(resume=...)`

**Learning goal (M3):** human-in-the-loop is not a UI feature — it is a graph primitive.
`interrupt()` suspends execution at any point, the checkpointer persists state, and
`Command(resume=value)` restores it. Step-07 replaces `MemorySaver` with
`AsyncPostgresSaver` so this works across process restarts.

## Setup

```bash
cp .env.example .env
uv sync
uv run python scripts/generate_data.py   # if not done already
```

## Run

```bash
uv run cloudops
```

## Test

```bash
uv run pytest tests/unit/
```

## What changed

| File | Change |
|---|---|
| `core/state.py` | New — `AgentState` TypedDict with `messages` |
| `agents/react.py` | New `ask_human` tool + `MemorySaver` checkpointer |
| `cli/run.py` | `_run_turn()` handles `GraphInterrupt` recursively |

## What to observe

1. **Trigger HITL:** ask an ambiguous question, e.g.:
   `"Show me incidents for the API service"` — the agent may ask
   `"Which datacenter? DC-A, DC-B, or DC-C?"` before querying.

2. **Watch the flow:** the agent calls `ask_human(question=...)`, execution suspends,
   the CLI prints the question, you answer, `Command(resume=answer)` replays the graph
   from the interrupt point. No data is lost.

3. **Compare with step-02:** ask the same question without HITL — the agent guesses.
   With HITL it confirms. This is the "probabilistic wall" from M3: for regulated or
   high-stakes queries, guessing is not acceptable.
