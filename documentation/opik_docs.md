# OPIK Comprehensive Documentation

> **Versions at time of writing:** `opik==2.0.41` (latest stable, released May 19 2026); `opik-optimizer==3.1.0` (latest stable, released Feb 24 2026).
> The user's installed versions are `opik==1.7.29` and `opik-optimizer==0.7.0`. All API surfaces described here apply to both versions unless noted.

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [Opik Client](#2-opik-client)
3. [LangChain Integration — OpikTracer](#3-langchain-integration--opiktracer)
4. [@opik.track Decorator](#4-opiktrack-decorator)
5. [Threads — Grouping Traces](#5-threads--grouping-traces)
6. [Prompt Management](#6-prompt-management)
7. [Experiments and Evaluation](#7-experiments-and-evaluation)
8. [Metrics](#8-metrics)
9. [Datasets](#9-datasets)
10. [opik-optimizer](#10-opik-optimizer)

---

## 1. Installation and Setup

### Install

```bash
pip install opik
pip install opik-optimizer          # separate package for prompt optimisation
pip install langchain langchain_openai  # only if using LangChain integration
```

### Cloud configuration (interactive)

```bash
opik configure
# Prompts for API key and workspace, saves to ~/.opik.config
```

### Programmatic configuration

```python
import opik

opik.configure(
    api_key="YOUR_OPIK_API_KEY",
    workspace="my-workspace",          # Opik Cloud workspace name
    force=True                         # overwrite existing config file
)
```

`configure()` full signature:

```python
opik.configure(
    api_key: str | None = None,
    workspace: str | None = None,
    url: str | None = None,       # for self-hosted instances
    use_local: bool = False,
    force: bool = False,
    automatic_approvals: bool = False
) -> None
```

### Environment variables

| Variable | Purpose |
|---|---|
| `OPIK_API_KEY` | API key for Opik Cloud authentication |
| `OPIK_WORKSPACE` | Workspace name (Opik Cloud) |
| `OPIK_PROJECT_NAME` | Default project name for traces / experiments |
| `OPIK_URL_OVERRIDE` | Override the server URL (self-hosted) |

These override values in `~/.opik.config`. Setting them is the recommended approach in CI/CD and containerised deployments.

```bash
export OPIK_API_KEY="sk-..."
export OPIK_WORKSPACE="acme-corp"
export OPIK_PROJECT_NAME="my-llm-app"
```

### Local (self-hosted) setup

```bash
# Launch the Opik server locally via Docker Compose
# then configure the SDK to point at it
opik configure --local
# or in code:
opik.configure(use_local=True)
```

---

## 2. Opik Client

The `opik.Opik` class is the main entry point for direct SDK interactions (creating traces, datasets, prompts, experiments).

### Constructor

```python
import opik

client = opik.Opik(
    project_name="my-project",   # defaults to "Default Project"
    workspace="my-workspace",    # uses configured workspace if omitted
    host="https://www.comet.com/opik/api",  # server URL
    api_key="sk-..."             # ignored for local deployments
)
```

### Key methods — Traces & Spans

| Method | Description |
|---|---|
| `client.trace(name, input, output, metadata, tags, feedback_scores, thread_id)` | Create and log a trace manually |
| `client.update_trace(trace_id, ...)` | Update an existing trace |
| `client.search_traces(filter_string, ...)` | Search traces using OQL |
| `client.get_trace_content(trace_id)` | Retrieve full trace data |
| `client.span(trace_id, name, input, output, model, provider, ...)` | Create a span inside a trace |
| `client.update_span(span_id, ...)` | Update an existing span |
| `client.search_spans(filter_string, trace_id, ...)` | Search spans |
| `client.get_span_content(span_id)` | Retrieve full span data |

### Key methods — Feedback

```python
client.log_traces_feedback_scores(
    scores=[{"id": trace_id, "name": "quality", "value": 0.9, "reason": "Good answer"}]
)
client.log_spans_feedback_scores(scores=[...])
client.log_threads_feedback_scores(scores=[...])
client.delete_trace_feedback_score(trace_id=..., name="quality")
```

### Key methods — Datasets

```python
dataset = client.create_dataset(name="qa-v1", description="Q&A dataset")
dataset = client.get_dataset(name="qa-v1")
dataset = client.get_or_create_dataset(name="qa-v1")
client.delete_dataset(dataset_id=dataset.id)
```

### Key methods — Experiments

```python
experiment = client.create_experiment(name="exp-001", dataset_name="qa-v1")
experiment = client.get_experiment_by_name(name="exp-001")
experiment = client.get_experiment_by_id(experiment_id="...")
experiments = client.get_dataset_experiments(dataset_id=dataset.id)
```

### Key methods — Prompts

```python
prompt = client.create_prompt(name="summariser", prompt="Summarise: {{text}}")
prompt = client.get_prompt(name="summariser", commit="abc123")  # specific version
history = client.get_prompt_history(name="summariser")          # all versions
results = client.search_prompts(filter_string="name = 'summariser'")
```

### Lifecycle

```python
client.flush()   # ensure all queued messages are sent (with timeout)
client.end()     # flush and close the session
```

### Utility

```python
url = client.get_project_url()           # project URL without HTTP call
auth_ok = client.auth_check()            # validate API key
rest = client.rest_client                # low-level REST client (not stable API)
```

---

## 3. LangChain Integration — OpikTracer

`OpikTracer` is a LangChain callback that automatically captures inputs, outputs, cost, token usage, and metadata for every step in a LangChain chain or agent.

### Installation

```bash
pip install opik langchain langchain_openai
```

### Constructor signature

```python
from opik.integrations.langchain import OpikTracer

tracer = OpikTracer(
    project_name: str | None = None,         # Opik project to log traces into
    tags: list[str] | None = None,           # applied to every trace
    metadata: dict[str, Any] | None = None,  # stored on every trace
    thread_id: str | None = None,            # groups traces into a conversation thread
    graph: Graph | None = None,              # LangGraph Graph object (for graph definition tracing)
    distributed_headers: DistributedTraceHeadersDict | None = None,
    **kwargs: Any
)
```

### Basic usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from opik.integrations.langchain import OpikTracer

tracer = OpikTracer(
    project_name="customer-support",
    tags=["langchain", "production"],
    metadata={"version": "1.2"},
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support agent."),
    ("human", "{user_message}")
])
chain = prompt_template | llm

response = chain.invoke(
    {"user_message": "How do I reset my password?"},
    config={"callbacks": [tracer]}   # <-- pass tracer here
)
```

### Passing the tracer — three patterns

```python
# 1. Per-invoke (recommended for per-request control)
chain.invoke(input_dict, config={"callbacks": [tracer]})

# 2. Bound to the chain at construction
chain_with_tracer = chain.with_config({"callbacks": [tracer]})
chain_with_tracer.invoke(input_dict)

# 3. Global callback (traces everything in the process)
from langchain.callbacks import set_global_handler
set_global_handler("opik")   # uses default Opik configuration
```

### Multi-turn conversations with thread_id

```python
session_id = "user-42-session-789"

for user_message in conversation_turns:
    tracer = OpikTracer(
        project_name="chatbot",
        thread_id=session_id,       # groups all traces for this session
    )
    response = chain.invoke(
        {"user_message": user_message},
        config={"callbacks": [tracer]}
    )
```

### Streaming with cost tracking

```python
llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    stream_usage=True,      # required to capture cost in streaming mode
)
```

### Accessing created traces

```python
traces = tracer.created_traces()
trace_ids = [t.id for t in traces]

# Update after the fact
for trace in traces:
    trace.update(tags=["reviewed"])
    trace.log_feedback_score(name="human-quality", value=1.0)

# Flush before script exit
tracer.flush()
```

### Hybrid: @opik.track + OpikTracer

```python
import opik
from opik.integrations.langchain import OpikTracer

@opik.track(project_name="hybrid-app")
def run_pipeline(user_input: str) -> str:
    tracer = OpikTracer()   # inherits parent trace context
    response = llm.invoke(user_input, config={"callbacks": [tracer]})
    return response.content
```

### Distributed tracing

```python
from opik.types import DistributedTraceHeadersDict

headers = DistributedTraceHeadersDict(
    opik_trace_id="trace-id-from-upstream-service",
    opik_parent_span_id="span-id-from-upstream"
)
tracer = OpikTracer(distributed_headers=headers)
```

---

## 4. @opik.track Decorator

Use `@opik.track` to instrument any Python function (sync, async, generator) so that its execution creates a trace or span in Opik automatically.

### Full signature

```python
opik.track(
    type: Literal['general', 'tool', 'llm', 'guardrail'] = 'general',
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    capture_input: bool = True,
    ignore_arguments: list[str] | None = None,
    capture_output: bool = True,
    generations_aggregator: Callable[[list[Any]], Any] | None = None,
    flush: bool = False,
    project_name: str | None = None
) -> Callable
```

| Parameter | Description |
|---|---|
| `type` | Span classification: `'general'` (default), `'tool'`, `'llm'`, `'guardrail'` |
| `tags` | Labels attached to this span |
| `metadata` | Arbitrary key-value data stored with the span |
| `capture_input` | Whether to record function arguments (default `True`) |
| `ignore_arguments` | List of argument names to exclude from recording |
| `capture_output` | Whether to record return value (default `True`) |
| `generations_aggregator` | Custom function to merge streaming output chunks |
| `flush` | If `True`, immediately flush the span to the server |
| `project_name` | Target project; overrides environment / config file |

### Usage patterns

```python
import opik

# Pattern 1: decorator without arguments
@opik.track
def my_function(prompt: str) -> str:
    return call_llm(prompt)

# Pattern 2: decorator with arguments
@opik.track(type='llm', tags=['gpt-4o'], project_name='my-project')
def call_model(prompt: str) -> str:
    return openai_client.chat.completions.create(...)

# Pattern 3: async function
@opik.track
async def async_pipeline(user_input: str) -> str:
    result = await async_llm_call(user_input)
    return result
```

### Nested functions (automatic parent-child spans)

```python
@opik.track
def retrieve_context(query: str) -> list[str]:
    return vector_db.search(query)

@opik.track
def generate_answer(query: str, context: list[str]) -> str:
    return llm.complete(f"Context: {context}\nQuery: {query}")

@opik.track                     # this becomes the root trace
def rag_pipeline(query: str) -> str:
    context = retrieve_context(query)   # child span
    return generate_answer(query, context)  # child span
```

### Hiding sensitive arguments

```python
@opik.track(ignore_arguments=["api_key", "user_pii"])
def process_request(query: str, api_key: str, user_pii: dict) -> str:
    return llm.complete(query)
```

### Accessing and modifying the current context inside a tracked function

```python
import opik
from opik import opik_context

@opik.track
def my_llm_call(prompt: str) -> str:
    response = llm.complete(prompt)

    # Enrich the current span with LLM-specific metadata
    opik_context.update_current_span(
        name="gpt-4o-call",
        model="gpt-4o",
        provider="openai",
        usage={"prompt_tokens": 50, "completion_tokens": 120, "total_tokens": 170},
        metadata={"temperature": 0.7},
    )

    # Enrich the root trace
    opik_context.update_current_trace(
        tags=["production"],
        metadata={"pipeline_version": "2.1"},
    )

    return response.text
```

#### opik_context functions

```python
from opik import opik_context

# Read current context
span_data  = opik_context.get_current_span_data()
trace_data = opik_context.get_current_trace_data()

# Update current span — full signature
opik_context.update_current_span(
    name: str | None,
    input: dict | None,
    output: dict | None,
    metadata: dict | None,
    tags: list[str] | None,
    usage: dict | OpikUsage | None,      # {"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}
    feedback_scores: list[FeedbackScoreDict] | None,
    model: str | None,
    provider: LLMProvider | str | None,
    total_cost: float | None,
    attachments: list[Attachment] | None,
    error_info: ErrorInfoDict | None,
    prompts: list[Prompt] | None
)

# Update current trace — full signature
opik_context.update_current_trace(
    name: str | None,
    input: dict | None,
    output: dict | None,
    metadata: dict | None,
    tags: list[str] | None,
    feedback_scores: list[FeedbackScoreDict] | None,
    thread_id: str | None,              # <-- set thread_id here
    attachments: list[Attachment] | None,
    prompts: list[Prompt] | None
)
```

### Context managers (manual span control)

```python
import opik

with opik.start_as_current_trace("my-trace", input={"q": query}) as trace:
    with opik.start_as_current_span("retrieval", type="tool") as span:
        context = vector_db.search(query)
        span.update(output={"docs": context})
    answer = llm.complete(query, context)
    trace.update(output={"answer": answer})
```

---

## 5. Threads — Grouping Traces

A **thread** is a named sequence of traces that belong to a single logical conversation or workflow session. All traces sharing the same `thread_id` are grouped in the Opik UI as a single thread.

### Rules

- `thread_id` is a **user-defined string**, unique per project.
- Any trace can be assigned a `thread_id` at creation time or updated later.

### Via @opik.track + opik_context

```python
import opik
from opik import opik_context

@opik.track(project_name="chatbot")
def handle_message(user_id: str, session_id: str, message: str) -> str:
    response = llm.complete(message)

    # Assign the trace to a thread
    opik_context.update_current_trace(
        thread_id=f"user-{user_id}-session-{session_id}"
    )
    return response

# All calls from the same session are grouped into one thread
handle_message("u1", "s42", "Hello!")
handle_message("u1", "s42", "What can you do?")
```

### Via OpikTracer (LangChain)

```python
tracer = OpikTracer(
    project_name="chatbot",
    thread_id="user-42-session-789",
)
chain.invoke({"message": "Hello"}, config={"callbacks": [tracer]})
```

### Via Opik client directly

```python
client = opik.Opik(project_name="chatbot")
trace = client.trace(
    name="turn-1",
    input={"message": "Hello"},
    output={"reply": "Hi there!"},
    thread_id="user-42-session-789",
)
```

### evaluate_threads

Thread-level evaluation is supported via `opik.evaluate_threads()`, which scores entire threads using metrics like `ConversationalCoherenceMetric` and `UserFrustrationMetric`.

---

## 6. Prompt Management

Opik provides versioned prompt storage. Each time you create a `Prompt` with the same name but different text, a new version (commit) is recorded.

### Prompt class

```python
import opik

# Create or update a prompt (new version created automatically)
prompt = opik.Prompt(
    name="summariser-v2",
    prompt="Summarise the following text in {{max_sentences}} sentences:\n\n{{text}}",
    metadata={"author": "data-team", "language": "en"},
    # type=PromptType.MUSTACHE_   # default; supports {{variable}} syntax
)

print(prompt.name)    # "summariser-v2"
print(prompt.commit)  # SHA hash of this version
```

### Formatting (variable substitution)

```python
filled = prompt.format(text="Long article...", max_sentences=3)
# Returns: "Summarise the following text in 3 sentences:\n\nLong article..."
```

### Via Opik client

```python
client = opik.Opik()

# Create a new version
prompt = client.create_prompt(
    name="summariser-v2",
    prompt="Summarise: {{text}}"
)

# Retrieve the latest version
prompt = client.get_prompt(name="summariser-v2")

# Retrieve a specific version by commit hash
prompt = client.get_prompt(name="summariser-v2", commit="abc123def456")

# List all versions
history = client.get_prompt_history(name="summariser-v2")
for version in history:
    print(version.commit, version.prompt)

# Search prompts
results = client.search_prompts(filter_string="name = 'summariser-v2'")
```

### Linking prompts to traces / experiments

```python
@opik.track
def run_with_prompt(text: str) -> str:
    prompt = opik.Prompt(name="summariser-v2", prompt="Summarise: {{text}}")
    filled = prompt.format(text=text)
    response = llm.complete(filled)

    # Link the prompt version to this trace
    opik_context.update_current_trace(prompts=[prompt])
    return response
```

---

## 7. Experiments and Evaluation

Experiments let you run a task function over a dataset, score each output with metrics, and compare results across runs in the Opik UI.

### opik.evaluate()

```python
import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, AnswerRelevance

# Step 1: Get or create a dataset
client = opik.Opik()
dataset = client.get_or_create_dataset(name="qa-benchmark")

# Step 2: Insert test items
dataset.insert([
    {"input": "What is the capital of France?", "expected_output": "Paris"},
    {"input": "Who wrote Hamlet?",              "expected_output": "Shakespeare"},
])

# Step 3: Define a task function
# Must accept a dict (one dataset item) and return a dict
def my_task(dataset_item: dict) -> dict:
    response = llm.complete(dataset_item["input"])
    return {
        "output": response.text,
        "context": ["France is a country in Western Europe..."],   # for RAG metrics
    }

# Step 4: Run evaluation
result = evaluate(
    dataset=dataset,
    task=my_task,
    scoring_metrics=[
        Hallucination(),
        AnswerRelevance(),
    ],
    experiment_name="baseline-gpt4o",
    project_name="qa-eval",
    experiment_config={"model": "gpt-4o", "temperature": 0},
    nb_samples=100,          # evaluate only 100 items (None = all)
    task_threads=16,         # parallel workers
    trial_count=1,           # runs per item
    verbose=1,               # 0=silent, 1=summary, 2=detailed
)
```

#### evaluate() full signature

```python
opik.evaluation.evaluate(
    dataset: Dataset,
    task: Callable[[dict[str, Any]], dict[str, Any]],
    scoring_metrics: list[BaseMetric] | None = None,
    experiment_name: str | None = None,           # auto-generated if omitted
    project_name: str | None = None,
    experiment_config: dict[str, Any] | None = None,
    verbose: int = 1,
    nb_samples: int | None = None,
    task_threads: int = 16,
    prompt: Prompt | None = None,
    prompts: list[Prompt] | None = None,
    scoring_key_mapping: dict[str, str | Callable] | None = None,
    dataset_item_ids: list[str] | None = None,
    dataset_sampler: BaseDatasetSampler | None = None,
    trial_count: int = 1
) -> EvaluationResult
```

#### scoring_key_mapping

When the keys returned by your task don't match what the metric expects, use `scoring_key_mapping`:

```python
evaluate(
    ...,
    scoring_key_mapping={
        "reference": "expected_output",   # metric expects "reference", item has "expected_output"
        "context": lambda item: item.get("retrieved_docs", []),
    }
)
```

### evaluate_prompt()

Evaluate a prompt template directly without writing a task function. The prompt is rendered per dataset item using `{{variable}}` syntax.

```python
from opik.evaluation import evaluate_prompt

result = evaluate_prompt(
    dataset=dataset,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Answer the question: {{input}}"},
    ],
    model="gpt-4o",
    scoring_metrics=[Hallucination(), AnswerRelevance()],
    experiment_name="prompt-v3-eval",
    project_name="qa-eval",
    nb_samples=50,
)
```

### evaluate_experiment()

Re-score an existing experiment (e.g., with new metrics) without re-running the task:

```python
from opik.evaluation import evaluate_experiment

evaluate_experiment(
    experiment_name="baseline-gpt4o",
    scoring_metrics=[AnswerRelevance()],
)
```

---

## 8. Metrics

### Built-in heuristic metrics

| Class | Import | Description |
|---|---|---|
| `Equals` | `opik.evaluation.metrics` | Exact string match |
| `Contains` | `opik.evaluation.metrics` | Substring check |
| `RegexMatch` | `opik.evaluation.metrics` | Regex pattern match |
| `IsJson` | `opik.evaluation.metrics` | Valid JSON check |
| `LevenshteinRatio` | `opik.evaluation.metrics` | Edit distance ratio 0–1 |

```python
from opik.evaluation.metrics import LevenshteinRatio

metric = LevenshteinRatio()
result = metric.score(output="Paris, France", reference="Paris")
print(result.value)   # float between 0.0 and 1.0
print(result.name)    # "levenshtein_ratio_metric"
```

### Built-in LLM-as-a-judge metrics

| Class | Key score() arguments | Description |
|---|---|---|
| `Hallucination` | `input`, `output`, `context` | Detects hallucinations |
| `AnswerRelevance` | `input`, `output`, `context` | Answer relevance to the question |
| `ContextPrecision` | `input`, `output`, `expected_output`, `context` | RAG context precision |
| `ContextRecall` | `input`, `output`, `expected_output`, `context` | RAG context recall |
| `Moderation` | `input`, `output` | Content safety check |
| `GEval` | `output` (+ task_introduction, evaluation_criteria in constructor) | Flexible LLM judge |

#### Hallucination example

```python
from opik.evaluation.metrics import Hallucination

metric = Hallucination()
result = metric.score(
    input="What is the capital of France?",
    output="The capital of France is Berlin.",
    context=["France is a country. Its capital is Paris."]
)
print(result.value)   # 1.0 = hallucination detected, 0.0 = no hallucination
print(result.reason)
```

#### GEval example

```python
from opik.evaluation.metrics import GEval

metric = GEval(
    task_introduction="You are evaluating the quality of a customer support response.",
    evaluation_criteria="The response should be polite, accurate, and concise.",
    model="gpt-4o",
    name="support-quality",
    temperature=0.0,
    seed=42,
)
result = metric.score(output="Thank you for contacting us. Your issue has been resolved.")
print(result.value)   # float 0.0–1.0
```

#### ContextPrecision constructor

```python
from opik.evaluation.metrics import ContextPrecision

metric = ContextPrecision(
    model: str | OpikBaseModel | None = None,
    name: str = 'context_precision_metric',
    few_shot_examples: list | None = None,
    track: bool = True,
    project_name: str | None = None,
    seed: int | None = None,
    temperature: float | None = None
)
result = metric.score(
    input="query",
    output="answer",
    expected_output="ground truth",
    context=["doc1", "doc2"]
)
```

### ScoreResult

All metrics return a `ScoreResult`:

```python
from opik.evaluation.metrics.score_result import ScoreResult

ScoreResult(
    value: float,           # typically 0.0–1.0; higher = better
    name: str,              # metric name
    reason: str | None,     # optional explanation
    scoring_failed: bool    # True if metric could not score
)
```

### BaseMetric — Custom metric pattern

```python
from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult

class ExactMatchMetric(BaseMetric):
    def __init__(self, name: str = "exact_match", track: bool = True):
        super().__init__(name=name, track=track)

    def score(
        self,
        output: str,
        expected_output: str,
        **ignored_kwargs          # absorb extra keys from task output
    ) -> ScoreResult:
        match = output.strip().lower() == expected_output.strip().lower()
        return ScoreResult(
            value=1.0 if match else 0.0,
            name=self.name,
            reason=f"output={'matched' if match else 'did not match'} expected"
        )

    async def ascore(self, output: str, expected_output: str, **kwargs) -> ScoreResult:
        return self.score(output=output, expected_output=expected_output)
```

`BaseMetric` constructor parameters:

```python
BaseMetric(
    name: str | None = None,          # defaults to class name
    track: bool = True,               # whether to record metric calls as spans
    project_name: str | None = None   # project for standalone metric tracking
)
```

---

## 9. Datasets

Datasets are collections of items used in evaluations. They are managed server-side (persisted in Opik).

### Create or retrieve

```python
import opik

client = opik.Opik()

# Create (raises if already exists)
dataset = client.create_dataset(name="rag-test-v1", description="RAG evaluation set")

# Get existing
dataset = client.get_dataset(name="rag-test-v1")

# Create if not exists, get if it does
dataset = client.get_or_create_dataset(name="rag-test-v1")

# Delete
client.delete_dataset(dataset_id=dataset.id)
```

### Dataset properties

| Property | Type | Description |
|---|---|---|
| `id` | str | Unique identifier |
| `name` | str | Dataset name |
| `description` | str | Optional description |

### Insert items

Items are plain dicts. Use any keys you like — the keys become available in the task function and for `scoring_key_mapping`.

```python
dataset.insert([
    {
        "input": "What is the boiling point of water?",
        "expected_output": "100 degrees Celsius",
        "context": ["Water boils at 100°C at standard atmospheric pressure."]
    },
    {
        "input": "Who invented the telephone?",
        "expected_output": "Alexander Graham Bell",
    }
])
```

### Other insertion methods

```python
# From a JSONL file
dataset.read_jsonl_from_file(
    file_path="data.jsonl",
    keys_mapping={"q": "input", "a": "expected_output"},  # optional rename
    ignore_keys=["timestamp"]
)

# From a pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
dataset.insert_from_pandas(dataframe=df)

# From a JSON string
dataset.insert_from_json(
    json_array='[{"input": "Q1", "expected_output": "A1"}]'
)
```

### Retrieve and export

```python
items = dataset.get_items(nb_samples=50)   # list of dicts

df     = dataset.to_pandas()
json_s = dataset.to_json()
```

### Update and delete items

```python
# Update requires "id" field in each item
dataset.update([{"id": "item-id-123", "input": "Updated question"}])

# Delete specific items by ID
dataset.delete(items_ids=["item-id-123", "item-id-456"])

# Clear all items
dataset.clear()
```

---

## 10. opik-optimizer

`opik-optimizer` is a separate package that automates prompt engineering by iteratively refining prompts against a dataset and scoring function.

> **Latest stable:** `opik-optimizer==3.1.0` (Feb 2026). The user has `0.7.0` installed. Core concepts and APIs below apply to both; `3.x` added `GepaOptimizer`, `ParameterOptimizer`, optimizer chaining via `ChatPrompt.from_result()`, and native MCP tool support.

### Installation

```bash
pip install opik-optimizer
# Requires an LLM provider key
export OPENAI_API_KEY="sk-..."
# Optional: Opik tracing (log optimization runs to OPIK UI)
pip install opik && opik configure
```

### What it does — the MetaPrompt optimization loop

1. Start with an initial `ChatPrompt`.
2. Evaluate the prompt on a sample of the dataset using your scoring function.
3. Send the scored examples + current prompt to a "reasoning LLM" (the meta-prompter), which critiques the prompt and proposes improved candidates.
4. Evaluate the candidates.
5. Keep the best-performing candidate. Repeat for `max_trials` total evaluations.
6. Return an `OptimizationResult` with the best prompt, score history, and metadata.

If `project_name` is configured (or set on the optimizer), all runs, experiments, and prompt versions are logged to your Opik workspace automatically.

### ChatPrompt

`ChatPrompt` is the prompt container used by all optimizers.

```python
from opik_optimizer import ChatPrompt

# Simple form
prompt = ChatPrompt(
    system="You are a helpful assistant.",
    user="{question}"             # {variable} placeholders match dataset item keys
)

# Full messages list form (supports multimodal)
prompt = ChatPrompt(
    messages=[
        {"role": "system",    "content": "You are an expert assistant."},
        {"role": "user",      "content": "{question}"},
        # Multimodal user turn:
        # {"role": "user", "content": [
        #     {"type": "text",      "text": "{question}"},
        #     {"type": "image_url", "image_url": {"url": "{image_url}"}},
        # ]},
    ],
    model="gpt-4o-mini",          # model used during evaluation; can be set here
    project_name="optimizer-runs" # project for logging (preferred over optimizer kwarg in v3+)
)

# With tools (function calling)
prompt = ChatPrompt(
    system="You are a research assistant.",
    user="{question}",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "search_wikipedia",
                "description": "Search Wikipedia for a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }
    ],
    function_map={"search_wikipedia": my_search_fn}
)
```

### Scoring function signature

All optimizers use the same scoring function signature:

```python
def my_metric(dataset_item: dict, llm_output: str) -> float | ScoreResult:
    """
    Args:
        dataset_item: one item from the Dataset (all keys available)
        llm_output:   the string output produced by the prompt on this item
    Returns:
        float (0.0–1.0, higher = better) or a ScoreResult
    """
    ...
```

Example using a built-in metric:

```python
from opik.evaluation.metrics import LevenshteinRatio
from opik.evaluation.metrics.score_result import ScoreResult

levenshtein = LevenshteinRatio()

def levenshtein_metric(dataset_item: dict, llm_output: str) -> ScoreResult:
    return levenshtein.score(
        reference=dataset_item["expected_output"],
        output=llm_output
    )
```

Example with custom logic:

```python
def exact_match_metric(dataset_item: dict, llm_output: str) -> float:
    expected = dataset_item["expected_output"].strip().lower()
    actual   = llm_output.strip().lower()
    return 1.0 if expected == actual else 0.0
```

### MetaPromptOptimizer

```python
from opik_optimizer import MetaPromptOptimizer, ChatPrompt

optimizer = MetaPromptOptimizer(
    model="gpt-4o",             # LiteLLM model string for the meta-reasoning LLM
    n_threads=8,                # parallel evaluation threads (replaces deprecated num_threads)
    model_parameters={          # extra LiteLLM parameters for optimizer's own calls
        "temperature": 1.0,
        "max_tokens": 4096,
    }
)

result = optimizer.optimize_prompt(
    prompt=prompt,              # ChatPrompt
    dataset=dataset,            # opik Dataset or opik_optimizer built-in dataset
    metric=levenshtein_metric,  # scoring function
    n_samples=100,              # dataset items to evaluate per trial
    experiment_config={         # stored as metadata in OPIK
        "run_id": "run-001",
        "team": "ml-infra",
    },
)
```

### FewShotBayesianOptimizer

Uses Optuna (Bayesian optimization) to find the optimal number and combination of few-shot examples.

```python
from opik_optimizer import FewShotBayesianOptimizer, ChatPrompt
from opik_optimizer.datasets import hotpot   # built-in benchmark dataset

dataset = hotpot(count=300)

prompt = ChatPrompt(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "{question}"}
    ]
)

optimizer = FewShotBayesianOptimizer(
    model="gpt-4o-mini",     # model used to GENERATE the few-shot template
    min_examples=3,
    max_examples=8,
    n_threads=16,
    seed=42,
)

result = optimizer.optimize_prompt(
    prompt=prompt,
    dataset=dataset,
    metric=levenshtein_metric,
    n_samples=150,
)

result.display()
```

### Standardized optimize_prompt interface

All optimizers share:

```python
optimizer.optimize_prompt(
    prompt: ChatPrompt,
    dataset: Dataset,
    metric: Callable,                             # (dataset_item, llm_output) -> float | ScoreResult
    experiment_config: dict | None = None,
    n_samples: int | None = None,
    auto_continue: bool = False,                  # resume a previous run
    agent_class: type[OptimizableAgent] | None = None,
    **kwargs                                      # optimizer-specific extra params
) -> OptimizationResult
```

### OptimizationResult

```python
result = optimizer.optimize_prompt(...)

result.display()                  # pretty-print best prompt + score

best_prompt = result.best_prompt  # ChatPrompt with the winning messages
best_score  = result.best_score   # float

# Optimizer chaining: use best result as starting point for another optimizer
from opik_optimizer import MetaPromptOptimizer

prompt2 = ChatPrompt.from_result(result)   # v3+ API
optimizer2 = MetaPromptOptimizer(model="gpt-4o")
result2 = optimizer2.optimize_prompt(prompt=prompt2, dataset=dataset, metric=levenshtein_metric)
```

### Available optimizers summary

| Class | Algorithm | Best for |
|---|---|---|
| `MetaPromptOptimizer` | LLM-as-optimizer (meta-prompting) | General prompt refinement |
| `FewShotBayesianOptimizer` | Bayesian (Optuna) + few-shot search | Finding optimal examples |
| `EvolutionaryOptimizer` | Genetic algorithm | Large search spaces |
| `GepaOptimizer` | Genetic-Pareto (multi-objective) | Balancing competing goals |
| `ParameterOptimizer` | Bayesian param tuning | Optimising temperature, top_p, etc. |

### Tool optimization (beta, v3+)

```python
prompt = ChatPrompt(
    system="You are a research assistant.",
    user="{question}",
    tools=[{"type": "function", "function": {...}}],
)

result = optimizer.optimize_prompt(
    prompt=prompt,
    dataset=dataset,
    metric=my_metric,
    optimize_tools=True,              # optimize tool descriptions (not schemas)
    # optimize_tools={"search_wikipedia": True}   # specific tools only
)
```

### MCP tools (remote server, v3+)

```python
prompt = ChatPrompt(
    system="You are a documentation assistant.",
    user="{question}",
    tools=[
        {
            "type": "mcp",
            "server_label": "context7",
            "server_url": "https://mcp.context7.com/mcp",
            "headers": {"CONTEXT7_API_KEY": "$CONTEXT7_API_KEY"},
            "allowed_tools": ["resolve-library-id", "query-docs"]
        }
    ]
)
```

### How prompt versions are saved to OPIK

When you set `project_name` (either on the `ChatPrompt` or, in older versions, on the optimizer constructor), each optimization run is automatically logged:

- Every candidate prompt evaluated is recorded as an experiment in the configured Opik project.
- The final `OptimizationResult` is linked to a named experiment with metadata including optimizer version, tool schemas, model strings, and `experiment_config` contents.
- You can view, compare, and retrieve these prompt versions in the Opik UI under **Prompt Management** and **Experiments**.

```python
# Logging to OPIK (set project_name on ChatPrompt in v3+)
prompt = ChatPrompt(
    system="You are a helpful assistant.",
    user="{question}",
    project_name="prompt-optimization-runs"   # <-- traces logged here
)

result = optimizer.optimize_prompt(
    prompt=prompt,
    dataset=dataset,
    metric=my_metric,
    experiment_config={"run_tag": "sprint-42"},
)
# All trials are now visible in OPIK UI under "prompt-optimization-runs"
```

### Built-in benchmark datasets (opik_optimizer.datasets)

```python
from opik_optimizer.datasets import hotpot, driving_hazard

dataset_train = hotpot(count=300)       # HotpotQA subset
dataset_val   = hotpot(count=50)

dataset_hazard = driving_hazard(count=20)
```

---

## Quick Reference: Common Patterns

### Trace any function

```python
@opik.track(project_name="prod", type="llm", tags=["v2"])
def my_llm_fn(prompt: str) -> str: ...
```

### LangChain chain with thread

```python
tracer = OpikTracer(project_name="chatbot", thread_id=session_id)
chain.invoke(input, config={"callbacks": [tracer]})
```

### Run an evaluation experiment

```python
dataset = client.get_or_create_dataset("my-ds")
dataset.insert([{"input": "...", "expected_output": "..."}])
evaluate(dataset=dataset, task=my_task, scoring_metrics=[Hallucination()], experiment_name="exp-1")
```

### Custom metric

```python
class MyMetric(BaseMetric):
    def score(self, output, expected_output, **kw) -> ScoreResult:
        return ScoreResult(value=..., name=self.name)
```

### Optimize a prompt

```python
optimizer = MetaPromptOptimizer(model="gpt-4o")
result = optimizer.optimize_prompt(prompt=prompt, dataset=dataset, metric=my_fn)
result.display()
```

---

## Sources

- [Opik Python SDK Reference](https://www.comet.com/docs/opik/python-sdk-reference/)
- [Opik Documentation Home](https://www.comet.com/docs/opik/)
- [opik on PyPI](https://pypi.org/project/opik/)
- [opik-optimizer on PyPI](https://pypi.org/project/opik-optimizer/)
- [opik GitHub repository](https://github.com/comet-ml/opik)
- [opik-optimizer README](https://github.com/comet-ml/opik/blob/main/sdks/opik_optimizer/README.md)
- [LangChain integration docs](https://www.comet.com/docs/opik/integrations/langchain)
- [OpikTracer API reference](https://www.comet.com/docs/opik/python-sdk-reference/integrations/langchain/OpikTracer.html)
- [track decorator reference](https://www.comet.com/docs/opik/python-sdk-reference//track.html)
- [evaluate() reference](https://www.comet.com/docs/opik/python-sdk-reference/evaluation/evaluate.html)
- [Dataset reference](https://www.comet.com/docs/opik/python-sdk-reference/evaluation/Dataset.html)
- [BaseMetric reference](https://www.comet.com/docs/opik/python-sdk-reference/evaluation/metrics/BaseMetric.html)
- [GEval reference](https://www.comet.com/docs/opik/python-sdk-reference/evaluation/metrics/GEval.html)
- [ContextPrecision reference](https://www.comet.com/docs/opik/python-sdk-reference/evaluation/metrics/ContextPrecision.html)
- [update_current_trace reference](https://www.comet.com/docs/opik/python-sdk-reference/opik_context/update_current_trace.html)
- [update_current_span reference](https://www.comet.com/docs/opik/python-sdk-reference/opik_context/update_current_span.html)
- [Opik Changelog March 2026](https://www.comet.com/docs/opik/changelog/2026/3/3)
- [MetaPrompt optimizer docs](https://www.comet.com/docs/opik/agent_optimization/algorithms/metaprompt_optimizer)
- [FewShot Bayesian optimizer docs](https://www.comet.com/docs/opik/agent_optimization/algorithms/fewshot_bayesian_optimizer)
