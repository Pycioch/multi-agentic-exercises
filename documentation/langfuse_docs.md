# Langfuse Python SDK v4 — Comprehensive Reference

**Installed version**: `langfuse==4.6.1` (latest stable as of May 2026)  
**Python requirements**: `>=3.10, <4.0`  
**Pydantic requirement**: v2 (v1 dropped in v4)

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Langfuse Client](#2-langfuse-client)
3. [LangChain Integration](#3-langchain-integration)
4. [Tracing — Traces, Spans, Generations](#4-tracing--traces-spans-generations)
5. [Sessions](#5-sessions)
6. [Prompt Management](#6-prompt-management)
7. [Datasets](#7-datasets)
8. [Experiments / run_experiment](#8-experiments--run_experiment)
9. [Scores](#9-scores)
10. [Flushing](#10-flushing)
11. [Decorators — @observe](#11-decorators--observe)
12. [Key Imports Cheatsheet](#12-key-imports-cheatsheet)

---

## 1. Installation & Setup

### Install

```bash
pip install langfuse
# With LangChain support:
pip install langfuse langchain langchain_openai langgraph
```

### Environment Variables

Set these before importing Langfuse (or pass as constructor arguments):

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # EU region (default)
# For US region:
# export LANGFUSE_HOST="https://us.cloud.langfuse.com"
# For self-hosted:
# export LANGFUSE_HOST="https://your-langfuse-instance.com"
```

> **Note**: `LANGFUSE_BASE_URL` is also accepted as an alias for `LANGFUSE_HOST`.

Optional environment variables:
```bash
export LANGFUSE_DEBUG="True"          # enable debug logging
export LANGFUSE_SAMPLE_RATE="0.5"    # send only 50% of traces
```

### Verify Connection

```python
from langfuse import get_client

langfuse = get_client()
langfuse.auth_check()  # raises if credentials are invalid
```

---

## 2. Langfuse Client

### Initialization

**Recommended — singleton via `get_client()`** (reads env vars automatically):

```python
from langfuse import get_client

langfuse = get_client()
```

**Direct constructor** (useful for multi-project setups or explicit config):

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
    debug=False,
    sample_rate=1.0,   # 0.0–1.0
)
```

`get_client()` returns the same singleton every time — safe to call from anywhere.

### Key Client Methods

| Method | Description |
|--------|-------------|
| `get_client()` | Returns the global singleton Langfuse client |
| `langfuse.auth_check()` | Verifies credentials and connectivity |
| `langfuse.flush()` | Blocks until all queued events are sent |
| `langfuse.shutdown()` | Flushes then shuts down background worker |
| `langfuse.start_as_current_observation(...)` | Context manager — creates a span/generation |
| `langfuse.create_score(...)` | Attaches a score to a trace or observation |
| `langfuse.create_prompt(...)` | Creates or versions a prompt |
| `langfuse.get_prompt(...)` | Fetches a prompt (with caching) |
| `langfuse.create_dataset(...)` | Creates a new dataset |
| `langfuse.create_dataset_item(...)` | Adds an item to a dataset |
| `langfuse.get_dataset(...)` | Fetches a dataset with all its items |
| `langfuse.run_experiment(...)` | Runs an experiment against local data |

---

## 3. LangChain Integration

### Setup

```python
from langfuse import get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()
langfuse_handler = CallbackHandler()
```

### Passing the Handler to Chains

Use the `config={"callbacks": [...]}` pattern on any LangChain runnable:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | model

# Sync invocation
result = chain.invoke(
    {"topic": "cats"},
    config={"callbacks": [langfuse_handler]}
)

# Async invocation
result = await chain.ainvoke(
    {"topic": "dogs"},
    config={"callbacks": [langfuse_handler]}
)

# Batch
results = chain.batch(
    [{"topic": "cats"}, {"topic": "dogs"}],
    config={"callbacks": [langfuse_handler]}
)

# Streaming
for chunk in chain.stream(
    {"topic": "birds"},
    config={"callbacks": [langfuse_handler]}
):
    print(chunk.content, end="")
```

### Setting Trace Attributes via Metadata

Pass `session_id`, `user_id`, and `tags` dynamically through config metadata:

```python
result = chain.invoke(
    {"topic": "cats"},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": "session-abc-123",
            "langfuse_user_id": "user-456",
            "langfuse_tags": ["production", "chat-feature"],
            "langfuse_trace_name": "my-custom-trace-name",
        }
    }
)
```

### Interoperability with Native SDK

Combine `@observe` decorator with `CallbackHandler` to nest LangChain traces inside custom spans:

```python
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

@observe()
def handle_user_message(user_input: str, session_id: str):
    langfuse = get_client()

    with propagate_attributes(
        trace_name="handle-user-message",
        session_id=session_id,
        user_id="user-789",
        tags=["v2"],
    ):
        langfuse_handler = CallbackHandler()
        result = chain.invoke(
            {"input": user_input},
            config={"callbacks": [langfuse_handler]}
        )
    return result
```

Using context managers directly:

```python
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span",
    name="multi-step-pipeline"
) as root_span:
    with propagate_attributes(session_id="session-123", user_id="user-456"):
        langfuse_handler = CallbackHandler()
        result = chain.invoke(
            {"input": "some input"},
            config={"callbacks": [langfuse_handler]}
        )
    root_span.update(output=result)
```

---

## 4. Tracing — Traces, Spans, Generations

### Concepts

- **Trace**: Top-level container for one request/interaction. Created automatically when you start an observation.
- **Span**: A unit of work inside a trace (retrieval, preprocessing, tool call, etc.).
- **Generation**: A specialized span representing an LLM call — captures model, input prompt, output, token counts, latency.

In LangChain integration:
- Each `chain.invoke()` → one **Trace**
- Each chain step (prompt, retriever, tool) → nested **Span**
- Each LLM call → **Generation** with prompt/completion details

### Low-Level Tracing via Context Managers

```python
from langfuse import get_client

langfuse = get_client()

# Create a span (unit of work)
with langfuse.start_as_current_observation(
    as_type="span",
    name="document-retrieval",
    input={"query": "What is RAG?"},
    metadata={"source": "vector-db"},
) as span:
    docs = retrieve_documents("What is RAG?")
    span.update(output={"docs_count": len(docs)})

# Create a generation (LLM call)
with langfuse.start_as_current_observation(
    as_type="generation",
    name="answer-generation",
    model="gpt-4o",
    input=[{"role": "user", "content": "Summarize these docs"}],
) as generation:
    response = call_llm(...)
    generation.update(
        output=response.content,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        }
    )
```

### Nesting Observations

Observations nest automatically when created inside an active context:

```python
with langfuse.start_as_current_observation(as_type="span", name="parent") as parent:
    # This span is a child of "parent"
    with langfuse.start_as_current_observation(as_type="span", name="child") as child:
        child.update(output="child result")
    parent.update(output="parent result")
```

### Updating Observation Attributes

```python
with langfuse.start_as_current_observation(as_type="span", name="my-span") as span:
    result = do_work()
    span.update(
        output=result,
        metadata={"latency_ms": 120},
        level="DEFAULT",   # DEFAULT | DEBUG | WARNING | ERROR
    )
```

---

## 5. Sessions

Sessions group multiple traces that belong to the same conversation or user journey.

### With LangChain (via metadata)

```python
SESSION_ID = "conversation-xyz-789"

# Trace 1 — turn 1
chain.invoke(
    {"input": "Hello!"},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {"langfuse_session_id": SESSION_ID}
    }
)

# Trace 2 — turn 2 (same session)
chain.invoke(
    {"input": "What did I just say?"},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {"langfuse_session_id": SESSION_ID}
    }
)
```

### With Native SDK

```python
from langfuse import get_client, propagate_attributes

langfuse = get_client()

with propagate_attributes(session_id="conversation-xyz-789", user_id="user-42"):
    with langfuse.start_as_current_observation(as_type="span", name="turn-1") as span:
        result = process_turn_1()
        span.update(output=result)
```

All traces sharing the same `session_id` appear grouped in the Langfuse Sessions view.

---

## 6. Prompt Management

### Create a Prompt

```python
from langfuse import get_client

langfuse = get_client()

# Text prompt (single string template)
langfuse.create_prompt(
    name="movie-critic",
    type="text",
    prompt="As a {{criticlevel}} movie critic, do you like {{movie}}?",
    labels=["production"],         # promotes this version to production
    config={"temperature": 0.7},   # optional model config to store alongside prompt
)

# Chat prompt (list of messages)
langfuse.create_prompt(
    name="movie-critic-chat",
    type="chat",
    prompt=[
        {"role": "system", "content": "You are an expert {{criticlevel}} movie critic."},
        {"role": "user", "content": "Tell me your thoughts on {{movie}}."},
    ],
    labels=["production"],
)
```

Each call to `create_prompt` with an existing name creates a **new version** (auto-incremented).

### Retrieve a Prompt

```python
# Fetch "production" version (default)
prompt = langfuse.get_prompt("movie-critic")

# Fetch by label
prompt = langfuse.get_prompt("movie-critic", label="staging")

# Fetch specific version number
prompt = langfuse.get_prompt("movie-critic", version=3)

# Chat prompt — must specify type
chat_prompt = langfuse.get_prompt("movie-critic-chat", type="chat")
```

### Variable Substitution — `.compile()`

```python
prompt = langfuse.get_prompt("movie-critic")

# Substitute template variables
compiled_text = prompt.compile(criticlevel="expert", movie="Dune 2")
# Result: "As an expert movie critic, do you like Dune 2?"

# For chat prompts — returns list of message dicts
chat_prompt = langfuse.get_prompt("movie-critic-chat", type="chat")
compiled_messages = chat_prompt.compile(criticlevel="expert", movie="Dune 2")
# Result: [{"role": "system", "content": "You are an expert expert movie critic."}, ...]
```

### Use Compiled Prompt with LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

prompt_obj = langfuse.get_prompt("movie-critic-chat", type="chat")
messages = prompt_obj.compile(criticlevel="senior", movie="Oppenheimer")

model = ChatOpenAI(model="gpt-4o")
response = model.invoke(
    [SystemMessage(content=m["content"]) if m["role"] == "system"
     else HumanMessage(content=m["content"])
     for m in messages],
    config={"callbacks": [langfuse_handler]}
)
```

### Prompt Caching

Langfuse caches fetched prompts client-side. If you update a prompt in the UI and need the latest version immediately, either restart your application or use `cache_ttl_seconds=0`:

```python
prompt = langfuse.get_prompt("movie-critic", cache_ttl_seconds=0)
```

---

## 7. Datasets

Datasets are versioned collections of `(input, expected_output)` pairs used for evaluation.

### Create a Dataset

```python
langfuse.create_dataset(
    name="qa-evaluation",
    description="Q&A pairs for evaluating the support bot",
    metadata={"author": "alice", "version": "1.0"},
)
```

### Add Items

```python
langfuse.create_dataset_item(
    dataset_name="qa-evaluation",
    input={"question": "What is the refund policy?"},
    expected_output={"answer": "30-day full refund, no questions asked."},
    metadata={"source": "faq-doc"},
)

# Add multiple items
qa_pairs = [
    ("How do I reset my password?", "Click 'Forgot password' on the login page."),
    ("What payment methods do you accept?", "Visa, Mastercard, PayPal."),
]
for question, answer in qa_pairs:
    langfuse.create_dataset_item(
        dataset_name="qa-evaluation",
        input={"question": question},
        expected_output={"answer": answer},
    )
```

### Retrieve a Dataset

```python
dataset = langfuse.get_dataset("qa-evaluation")

# Access items
for item in dataset.items:
    print(item.input)            # {"question": "..."}
    print(item.expected_output)  # {"answer": "..."}
    print(item.id)               # unique item ID
```

### Dataset Versioning

Every add/update/delete/archive of items creates a new dataset version:

```python
# Fetch a specific point-in-time version (by timestamp string)
dataset = langfuse.get_dataset("qa-evaluation", version="2026-05-01T10:00:00Z")
```

### Folder Organization

Use slashes in the dataset name to organize into folders:

```python
langfuse.create_dataset(name="evaluation/support-bot/qa-pairs")
langfuse.create_dataset(name="evaluation/support-bot/edge-cases")
```

---

## 8. Experiments / run_experiment

Experiments test your application against a dataset (or local data) and attach scores. As of April 2026, experiments are a first-class concept in Langfuse.

### High-Level API: `run_experiment` (v4)

#### Against Local Data

```python
from langfuse import get_client, Evaluation

langfuse = get_client()

def my_task(*, item, **kwargs):
    """item is one element from the data list."""
    question = item["input"]["question"]
    answer = my_llm_app(question)      # your application call
    return answer

def accuracy_evaluator(*, input, output, expected_output, **kwargs):
    """Return an Evaluation or None."""
    if expected_output and expected_output["answer"].lower() in output.lower():
        return Evaluation(name="accuracy", value=1.0)
    return Evaluation(name="accuracy", value=0.0)

local_data = [
    {"input": {"question": "Capital of France?"}, "expected_output": {"answer": "Paris"}},
    {"input": {"question": "Capital of Germany?"}, "expected_output": {"answer": "Berlin"}},
]

result = langfuse.run_experiment(
    name="geography-quiz-v1",
    data=local_data,
    task=my_task,
    evaluators=[accuracy_evaluator],
    max_concurrency=4,
)
```

#### Against a Langfuse Dataset

```python
dataset = langfuse.get_dataset("qa-evaluation")

result = dataset.run_experiment(
    name="support-bot-v2",
    task=my_task,
    evaluators=[accuracy_evaluator],
    max_concurrency=4,
    metadata={"model": "gpt-4o", "prompt_version": 3},
)
```

### Task Function Signature

```python
def my_task(*, item, **kwargs):
    # item["input"] — the dataset item input
    # item["expected_output"] — the dataset item expected output (if set)
    result = call_your_app(item["input"])
    return result  # return value becomes the "output" scored by evaluators
```

### Evaluator Function Signature

```python
def my_evaluator(*, input, output, expected_output, **kwargs):
    # input — item input dict
    # output — what my_task returned
    # expected_output — item expected_output dict
    score = compute_score(output, expected_output)
    return Evaluation(name="my-metric", value=score)
    # Return None to skip scoring this item
```

### Using AutoEvals (Pre-built LLM Judges)

```python
from langfuse.experiment import create_evaluator_from_autoevals
from autoevals.llm import Factuality

factuality_evaluator = create_evaluator_from_autoevals(Factuality())

result = dataset.run_experiment(
    name="factuality-test",
    task=my_task,
    evaluators=[factuality_evaluator],
)
```

### Low-Level Manual Iteration (for fine-grained control)

```python
from langfuse import get_client

langfuse = get_client()
dataset = langfuse.get_dataset("qa-evaluation")
run_name = "manual-run-v1"

for item in dataset.items:
    with item.run(run_name=run_name) as root_span:
        output = my_llm_app(item.input["question"])
        root_span.update(output=output)
        root_span.score_trace(
            name="accuracy",
            value=1.0 if item.expected_output["answer"] in output else 0.0,
            data_type="NUMERIC",
        )

langfuse.flush()
```

---

## 9. Scores

Scores attach evaluation results to traces, observations, sessions, or dataset runs.

### Score Data Types

| Type | Values | Example |
|------|--------|---------|
| `NUMERIC` | float | `0.92`, `1.0`, `42` |
| `CATEGORICAL` | string | `"positive"`, `"negative"` |
| `BOOLEAN` | `True`/`False` | pass/fail |
| `TEXT` | string | freeform annotation |

### Create Score via SDK

```python
from langfuse import get_client

langfuse = get_client()

# Score by trace_id (obtained from a prior trace)
langfuse.create_score(
    trace_id="trace-uuid-...",
    name="user-feedback",
    value=1.0,
    data_type="NUMERIC",
    comment="The response was helpful",
)

# Categorical score
langfuse.create_score(
    trace_id="trace-uuid-...",
    name="sentiment",
    value="positive",
    data_type="CATEGORICAL",
)

# Score a specific observation (span/generation) within a trace
langfuse.create_score(
    trace_id="trace-uuid-...",
    observation_id="observation-uuid-...",
    name="faithfulness",
    value=0.87,
    data_type="NUMERIC",
)
```

### Score from Inside a Context Manager

```python
with langfuse.start_as_current_observation(as_type="span", name="my-span") as span:
    result = do_work()
    span.update(output=result)

    # Score the entire trace that this span belongs to
    span.score_trace(
        name="quality",
        value=1.0,
        data_type="NUMERIC",
        comment="Looks good",
    )
```

### LLM-as-a-Judge Evaluators

Configure LLM-as-judge in the Langfuse UI (Evaluation → LLM-as-a-Judge) to automatically score production traces. You define:
- **Criteria name**: e.g., `"helpfulness"`, `"toxicity"`
- **Prompt template**: instructions for the LLM judge
- **Output type**: NUMERIC, CATEGORICAL, or BOOLEAN
- **Sampling**: % of traces to evaluate automatically

For programmatic LLM-as-judge during experiments, use the `autoevals` library (see Section 8).

### Scores via Annotation Queues

In the UI: Evaluation → Annotation Queues. Assign traces to human reviewers who score them manually — useful for ground-truth labeling.

---

## 10. Flushing

Langfuse batches events and sends them asynchronously in the background. In **long-running services** this happens automatically. In **short-lived scripts** or **serverless functions**, you must flush manually to ensure events are delivered before the process exits.

### When to Call flush()

```python
from langfuse import get_client

langfuse = get_client()

# ... your LLM calls ...

# At end of script / Lambda handler / Cloud Function:
langfuse.flush()
```

### shutdown() — Flush + Clean Shutdown

```python
langfuse.shutdown()   # flushes then terminates background worker thread
```

Use `shutdown()` when you are done with Langfuse entirely (e.g., end of a batch job).

### flush() After LangChain Operations

```python
result = chain.invoke(
    {"input": "Hello"},
    config={"callbacks": [langfuse_handler]}
)

# Make sure trace is sent in short-lived processes
get_client().flush()
```

### Environment-Specific Guidance

| Environment | Recommendation |
|-------------|----------------|
| FastAPI / Flask long-running | Not needed — background worker handles it |
| AWS Lambda / Cloud Run | Call `langfuse.flush()` at end of handler |
| CLI scripts / notebooks | Call `langfuse.flush()` at end |
| Batch jobs | Call `langfuse.shutdown()` at very end |

---

## 11. Decorators — @observe

The `@observe()` decorator instruments Python functions without modifying their internal logic. It automatically captures inputs, outputs, timing, and errors.

### Basic Usage

```python
from langfuse import observe, get_client

@observe()
def my_pipeline(user_query: str) -> str:
    # Inputs (user_query) and output (return value) captured automatically
    result = process(user_query)
    return result
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | function name | Custom name for the observation |
| `as_type` | str | `"span"` | Observation type: `"span"` or `"generation"` |
| `capture_input` | bool | `True` | Whether to capture function arguments |
| `capture_output` | bool | `True` | Whether to capture return value |

### Examples

```python
from langfuse import observe

# Default span
@observe()
def retrieve_documents(query: str) -> list:
    return vector_db.search(query)

# Custom name
@observe(name="llm-completion")
def call_llm(prompt: str) -> str:
    return openai_client.complete(prompt)

# Mark as a generation (LLM call)
@observe(as_type="generation")
def generate_answer(context: str, question: str) -> str:
    return llm.invoke(f"{context}\n\nQuestion: {question}")

# Disable I/O capture (for privacy or large payloads)
@observe(capture_input=False, capture_output=False)
def process_pii_data(user_data: dict) -> dict:
    return anonymize(user_data)

# Async functions work the same way
@observe()
async def async_pipeline(query: str) -> str:
    result = await async_llm_call(query)
    return result
```

### Automatic Nesting

When decorated functions call other decorated functions, Langfuse automatically creates a parent-child trace hierarchy:

```python
@observe()
def child_step(data: str) -> str:
    return process(data)

@observe()
def parent_pipeline(input: str) -> str:
    step1 = child_step(input)       # child_step appears nested under parent_pipeline
    step2 = child_step(step1)
    return step2
```

### Updating Trace Attributes Inside @observe

```python
from langfuse import observe, get_client, propagate_attributes

@observe()
def my_function(query: str, session_id: str) -> str:
    with propagate_attributes(
        session_id=session_id,
        user_id="user-123",
        tags=["my-tag"],
    ):
        result = process(query)
    return result
```

### Disable Globally

```bash
export LANGFUSE_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED="False"
```

---

## 12. Key Imports Cheatsheet

```python
# ── Core client ──────────────────────────────────────────────────────────────
from langfuse import get_client          # singleton client factory (recommended)
from langfuse import Langfuse            # direct constructor

# ── Tracing utilities ─────────────────────────────────────────────────────────
from langfuse import observe             # @observe decorator
from langfuse import propagate_attributes  # context manager for session/user attrs

# ── Experiments ───────────────────────────────────────────────────────────────
from langfuse import Evaluation          # return type from evaluator functions

# ── LangChain integration ─────────────────────────────────────────────────────
from langfuse.langchain import CallbackHandler

# ── Experiment evaluators (optional, requires: pip install autoevals) ─────────
from langfuse.experiment import create_evaluator_from_autoevals
from autoevals.llm import Factuality, Hallucination, ClosedQA
```

### Minimal Working Example (LangChain + Scoring)

```python
import os
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Environment variables set: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
#    LANGFUSE_HOST, OPENAI_API_KEY

# 2. Init
langfuse = get_client()
langfuse_handler = CallbackHandler()

# 3. Build chain
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("Answer concisely: {question}")
chain = prompt | model

# 4. Invoke with tracing + session
result = chain.invoke(
    {"question": "What is the capital of Poland?"},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": "demo-session-1",
            "langfuse_user_id": "demo-user",
            "langfuse_tags": ["demo"],
        }
    }
)

print(result.content)

# 5. Score the trace
trace_id = langfuse_handler.get_trace_id()  # capture trace ID after invoke
langfuse.create_score(
    trace_id=trace_id,
    name="correctness",
    value=1.0,
    data_type="NUMERIC",
    comment="Warsaw — correct!",
)

# 6. Flush (required in scripts)
langfuse.flush()
```

---

## Quick Reference: v3 → v4 Breaking Changes

| v3 | v4 |
|----|----|
| `start_span()` | `start_as_current_observation(as_type="span")` |
| `start_generation()` | `start_as_current_observation(as_type="generation")` |
| `update_current_trace()` | `propagate_attributes()` context manager |
| `DatasetItemClient.run()` | Removed — use `item.run()` context manager |
| `api.observations_v_2` | `api.observations` |
| `api.score_v_2` | `api.scores` |
| Pydantic v1 | Pydantic v2 required |
| Metadata: any dict | Metadata: `dict[str, str]`, max 200 chars per value |

---

*Sources: [Langfuse Docs](https://langfuse.com/docs), [PyPI langfuse 4.6.1](https://pypi.org/project/langfuse/), [LangChain Integration](https://langfuse.com/integrations/frameworks/langchain), [Datasets](https://langfuse.com/docs/evaluation/experiments/datasets), [Experiments via SDK](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk), [Prompt Management](https://langfuse.com/docs/prompts/get-started), [Python v3→v4 Migration](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4)*
