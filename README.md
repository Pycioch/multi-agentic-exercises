# Multi-Agentic Exercises

Hands-on LangGraph workshop repository with 24 progressive exercises (`00` to `23`)
plus supporting code showcases and cloudops tracks.

## What is inside

- `exercises/` - core exercise files used during the workshop.
- `TASKS.md` - detailed task sheet describing what to build in each exercise.
- `code_showcases/` - standalone demos for selected patterns.
- `cloudops_exercises/` - separate cloudops-oriented exercise track.
- `.env.example` - environment variable template.

## Quick start

```bash
cd multi-agentic-exercises
python -m venv .venv
source .venv/bin/activate
pip install langgraph langchain-openai langchain-core langfuse python-dotenv
cp .env.example .env
# fill in OPENAI_API_KEY and LANGFUSE_* keys in .env
```

## Run exercises

- Start from `exercises/00_simple_langchain_invocation/00_simple_langchain_invocation.py`.
- Then continue in order from `exercises/01_*` to `exercises/23_*`.
- Full requirements and expected graph behavior are documented in `TASKS.md`.

Example run:

```bash
python exercises/01_sequential_pipeline/exercise.py
```