# Ragas — Comprehensive Documentation Reference

> **Current stable version: v0.4**
> Official docs: https://docs.ragas.io/en/stable/
> PyPI: https://pypi.org/project/ragas/

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [Core Concepts](#2-core-concepts)
3. [Dataset — EvaluationDataset, SingleTurnSample, MultiTurnSample](#3-dataset)
4. [Built-in Metrics](#4-built-in-metrics)
5. [evaluate() Function](#5-evaluate-function)
6. [LLM Configuration](#6-llm-configuration)
7. [LangChain Integration](#7-langchain-integration)
8. [Results — EvaluationResult](#8-results)
9. [Custom Metrics](#9-custom-metrics)
10. [Key Imports Cheatsheet](#10-key-imports-cheatsheet)

---

## 1. Installation and Setup

### Basic installation

```bash
pip install ragas
```

### Quickstart scaffold (optional)

```bash
# With uvx (recommended)
uvx ragas quickstart rag_eval
cd rag_eval
uv sync

# With pip
pip install ragas
ragas quickstart rag_eval
cd rag_eval
uv sync
```

The quickstart generates:

```
rag_eval/
├── evals.py               # evaluation workflow
├── rag.py                 # your RAG/LLM application
├── evals/
│   ├── datasets/          # test data files
│   ├── experiments/       # results saved as CSV
│   └── logs/              # execution logs
```

### Environment variables

```bash
# OpenAI (default evaluator)
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="..."
```

---

## 2. Core Concepts

Ragas is an **LLM evaluation framework** that moves from "vibe checks" to systematic evaluation loops for RAG pipelines and LLM applications.

### What Ragas evaluates

Ragas measures the performance of **Retrieval Augmented Generation (RAG)** systems along these axes:

| Axis | Question answered |
|------|-------------------|
| **Retrieval quality** | Did the retriever find the right chunks? (Context Precision, Context Recall) |
| **Generation faithfulness** | Does the answer only use information from retrieved context? (Faithfulness) |
| **Response relevance** | Does the answer actually address the user's question? (Response Relevancy) |
| **Factual correctness** | Is the answer factually correct compared to a reference? (FactualCorrectness) |
| **Semantic similarity** | How semantically close is the answer to a reference? (SemanticSimilarity) |

### Key architectural concepts

- **Metrics** — functions that score a sample. Can be LLM-based (non-deterministic) or non-LLM (deterministic).
- **EvaluationDataset** — a typed, homogeneous list of `SingleTurnSample` or `MultiTurnSample` objects.
- **evaluate()** — the main function that runs all metrics over a dataset and aggregates results.
- **EvaluationResult** — the object returned by `evaluate()`, containing per-sample scores and aggregate statistics.
- **Judge LLM** — the LLM used to score samples (separate from the LLM being evaluated).

---

## 3. Dataset

### SingleTurnSample

Represents a single question–answer interaction.

```python
from ragas import SingleTurnSample

sample = SingleTurnSample(
    user_input="What is the capital of France?",
    retrieved_contexts=[
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe."
    ],
    response="The capital of France is Paris.",
    reference="Paris",
    reference_contexts=["Paris is the capital of France."],  # optional
    rubric={                                                  # optional
        "accuracy": "Correct",
        "completeness": "High"
    }
)
```

**Fields:**

| Field | Type | Required by | Description |
|-------|------|-------------|-------------|
| `user_input` | `str` | Most metrics | The user's query |
| `retrieved_contexts` | `list[str]` | Faithfulness, ContextPrecision, ContextRecall | Context chunks retrieved from the knowledge base |
| `response` | `str` | Faithfulness, ResponseRelevancy, FactualCorrectness | The LLM-generated answer |
| `reference` | `str` | ContextRecall, FactualCorrectness, SemanticSimilarity | Ground-truth / expected answer |
| `reference_contexts` | `list[str]` | Non-LLM ContextPrecision | Ground-truth context chunks (optional) |
| `rubric` | `dict[str, str]` | Rubric-based metrics | Evaluation criteria (optional) |

---

### MultiTurnSample

Represents a multi-turn conversational interaction (e.g., agentic workflows).

```python
from ragas import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

conversation = [
    HumanMessage(content="What's the weather like in Warsaw today?"),
    AIMessage(
        content="Let me check that for you.",
        tool_calls=[ToolCall(name="WeatherAPI", args={"location": "Warsaw"})]
    ),
    ToolMessage(content="Sunny, 22°C in Warsaw."),
    AIMessage(content="It is sunny and 22°C in Warsaw today.")
]

sample = MultiTurnSample(
    user_input=conversation,
    reference="Provide current weather in Warsaw to the user."
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `user_input` | `list[HumanMessage \| AIMessage \| ToolMessage]` | The full conversation history |
| `reference` | `str` | Expected outcome or ground-truth response |

**Message types:**

- `HumanMessage(content="...")` — user turn
- `AIMessage(content="...", tool_calls=[ToolCall(name="...", args={...})])` — assistant turn
- `ToolMessage(content="...")` — tool output

---

### EvaluationDataset

A typed container for a list of samples (all must be the same type).

```python
from ragas import EvaluationDataset, SingleTurnSample

samples = [
    SingleTurnSample(
        user_input="Who wrote Hamlet?",
        retrieved_contexts=["Hamlet is a play by William Shakespeare."],
        response="William Shakespeare wrote Hamlet.",
        reference="William Shakespeare"
    ),
    SingleTurnSample(
        user_input="What year did WWII end?",
        retrieved_contexts=["World War II ended in 1945."],
        response="World War II ended in 1945.",
        reference="1945"
    ),
]

dataset = EvaluationDataset(samples=samples)
```

**Alternative constructors:**

```python
# From a list of dicts
dataset = EvaluationDataset.from_list([
    {
        "user_input": "...",
        "retrieved_contexts": ["..."],
        "response": "...",
        "reference": "..."
    }
])

# From a Hugging Face dataset
dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
```

**Constraint:** All samples in a dataset must be the same type (`SingleTurnSample` or `MultiTurnSample`).

---

## 4. Built-in Metrics

> **Import note:** In Ragas v0.4, the canonical import path is `ragas.metrics.collections`. Importing from `ragas.metrics` still works but triggers a deprecation warning.

### 4.1 Faithfulness

**What it measures:** Factual consistency of the response with the retrieved contexts. Checks whether every claim in the response can be supported by the retrieved context.

**Algorithm:**
1. Extract all claims from the response.
2. For each claim, check if the retrieved context supports it.
3. Score = supported claims / total claims

**Score:** 0–1 (higher = more faithful, less hallucination)

**Required fields:** `user_input`, `response`, `retrieved_contexts`

**Constructor:**

```python
from ragas.metrics.collections import Faithfulness

metric = Faithfulness(
    llm=evaluator_llm,           # required: BaseRagasLLM instance
    name="faithfulness",         # default
    max_retries=1,               # default
)
```

**Usage:**

```python
score = await metric.ascore(
    user_input="When was the first Super Bowl?",
    response="The first Super Bowl was held on Jan 15, 1967.",
    retrieved_contexts=["The First AFL–NFL World Championship Game was played on January 15, 1967."]
)
# Returns MetricResult with .value (float) and .reason (str)
```

---

### 4.2 ResponseRelevancy (formerly AnswerRelevancy)

**What it measures:** How well the response addresses the user's question. Penalizes incomplete, off-topic, or unnecessarily verbose answers. Does **not** measure factual correctness.

**Algorithm:**
1. LLM generates N artificial questions from the response (default N=3).
2. Cosine similarity is computed between each generated question's embedding and the original `user_input` embedding.
3. Score = mean cosine similarity across N questions.

**Score:** 0–1 (higher = more relevant)

**Required fields:** `user_input`, `response`

**Constructor:**

```python
from ragas.metrics.collections import AnswerRelevancy  # legacy name
# or (preferred in v0.4):
from ragas.metrics.collections import ResponseRelevancy

metric = ResponseRelevancy(
    llm=evaluator_llm,           # required
    embeddings=evaluator_emb,    # required
    strictness=3,                # number of questions generated (default: 3)
    name="answer_relevancy",     # default
)
```

---

### 4.3 ContextPrecision (LLMContextPrecision)

**What it measures:** Whether the retriever ranks relevant chunks higher than irrelevant ones. Evaluates the quality of retrieval ordering.

**Algorithm:**
Mean Precision@K across all positions:

```
ContextPrecision@K = Σ(Precision@k × relevance_k) / total_relevant_items_in_top_K
```

Where `relevance_k` ∈ {0, 1} indicates whether item at rank k is relevant.

**Score:** 0–1 (higher = relevant chunks appear earlier in results)

**Required fields:** `user_input`, `retrieved_contexts`, `reference` (or `response`)

**Constructor:**

```python
from ragas.metrics.collections import ContextPrecision

metric = ContextPrecision(
    llm=evaluator_llm,           # required
    name="context_precision",    # default
    max_retries=1,               # default
)
```

**Variants:**

```python
# LLM-based (uses reference answer)
from ragas.metrics.collections import LLMContextPrecisionWithReference
# Non-LLM-based (uses reference_contexts directly)
from ragas.metrics.collections import NonLLMContextPrecisionWithReference
```

---

### 4.4 ContextRecall (LLMContextRecall)

**What it measures:** Whether all relevant information needed to answer the question was retrieved. Checks if information from the reference answer is present in retrieved contexts.

**Algorithm:**
1. Break the `reference` answer into individual claims.
2. For each claim, check if it can be attributed to any retrieved context.
3. Score = claims supported by context / total claims in reference

**Score:** 0–1 (higher = less information missed)

**Required fields:** `user_input`, `retrieved_contexts`, `reference`

**Constructor:**

```python
from ragas.metrics.collections import ContextRecall

metric = ContextRecall(
    llm=evaluator_llm,           # required
    name="context_recall",       # default
    max_retries=1,               # default
)
```

> **Note:** `LLMContextRecall` (from `ragas.metrics`) is the legacy name; it is deprecated and will be removed in v1.0. Use `ContextRecall` from `ragas.metrics.collections` instead.

---

### 4.5 FactualCorrectness (replaces AnswerCorrectness)

**What it measures:** Factual accuracy of the response compared to a reference answer. Combines factual recall and precision using NLI-based claim decomposition.

**Algorithm:**
- Decomposes both `response` and `reference` into atomic claims.
- Computes F1 score based on how many reference claims appear in the response and vice versa.

**Score:** 0–1

**Required fields:** `response`, `reference`

**Constructor:**

```python
from ragas.metrics.collections import FactualCorrectness

metric = FactualCorrectness(
    llm=evaluator_llm,           # required
    name="factual_correctness",  # default
    mode="f1",                   # "f1" | "precision" | "recall" (default: "f1")
)
```

> **Note:** In v0.1.x `AnswerCorrectness` combined factual + semantic similarity with weights. In v0.4, use `FactualCorrectness` for factual accuracy and `SemanticSimilarity` separately if needed.

---

### 4.6 SemanticSimilarity (replaces AnswerSimilarity)

**What it measures:** Semantic closeness of the generated `response` to the `reference` answer using embedding-based cosine similarity.

**Score:** 0–1

**Required fields:** `response`, `reference`

**Constructor:**

```python
from ragas.metrics.collections import SemanticSimilarity

metric = SemanticSimilarity(
    embeddings=evaluator_emb,    # required: BaseRagasEmbeddings instance
    name="semantic_similarity",  # default
    threshold=None,              # optional float cutoff
    is_cross_encoder=False,      # default
)
```

> **Note:** `AnswerSimilarity` from older versions maps to this in v0.4.

---

### 4.7 Other notable metrics

| Metric class | Import | Measures | Required fields |
|---|---|---|---|
| `NoiseSensitivity` | `ragas.metrics.collections` | Sensitivity to irrelevant context chunks | `user_input`, `response`, `reference`, `retrieved_contexts` |
| `ContextEntityRecall` | `ragas.metrics.collections` | Entity coverage in retrieved contexts vs reference | `reference`, `retrieved_contexts` |
| `FaithfulnesswithHHEM` | `ragas.metrics.collections` | Faithfulness using a local HHEM model (no LLM needed) | `response`, `retrieved_contexts` |
| `AspectCritic` | `ragas.metrics.collections` | Binary yes/no on a custom aspect (e.g. "is it polite?") | `user_input`, `response` |
| `SimpleCriteriaScore` | `ragas.metrics.collections` | Numeric score on a custom criterion | `user_input`, `response` |
| `BleuScore` | `ragas.metrics.collections` | BLEU score vs reference | `response`, `reference` |
| `RougeScore` | `ragas.metrics.collections` | ROUGE score vs reference | `response`, `reference` |

---

## 5. evaluate() Function

### Signature

```python
from ragas import evaluate

evaluate(
    dataset: Union[Dataset, EvaluationDataset],
    metrics: Optional[Sequence[Metric]] = None,
    llm: Optional[BaseRagasLLM | LangchainLLM] = None,
    embeddings: Optional[BaseRagasEmbeddings | LangchainEmbeddings] = None,
    experiment_name: Optional[str] = None,
    callbacks: Callbacks = None,
    run_config: Optional[RunConfig] = None,
    token_usage_parser: Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: Optional[int] = None,
    return_executor: bool = False,
    allow_nest_asyncio: bool = True,
) -> Union[EvaluationResult, Executor]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `EvaluationDataset` | required | The dataset to evaluate |
| `metrics` | `list[Metric]` | `None` | Metrics to run. If `None`, a default set is used |
| `llm` | `BaseRagasLLM` | `None` | Judge LLM for LLM-based metrics; overrides per-metric LLM |
| `embeddings` | `BaseRagasEmbeddings` | `None` | Embedding model for embedding-based metrics |
| `experiment_name` | `str` | `None` | Label for this evaluation run (used in results tracking) |
| `run_config` | `RunConfig` | `None` | Controls timeouts, retries, concurrency |
| `raise_exceptions` | `bool` | `False` | If `True`, raises errors instead of returning NaN |
| `column_map` | `dict[str, str]` | `None` | Rename columns when using a raw HF dataset (maps dataset field → ragas field) |
| `show_progress` | `bool` | `True` | Show tqdm progress bar |
| `batch_size` | `int` | `None` | Number of samples per batch |
| `token_usage_parser` | `TokenUsageParser` | `None` | Enables cost tracking |
| `return_executor` | `bool` | `False` | Returns `Executor` object (cancellable run) instead of waiting |

### Complete usage example

```python
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    ContextRecall,
    ContextPrecision,
    FactualCorrectness,
    ResponseRelevancy,
    SemanticSimilarity,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. Configure judge LLM and embeddings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# 2. Build the dataset
samples = [
    SingleTurnSample(
        user_input="Who invented the telephone?",
        retrieved_contexts=[
            "Alexander Graham Bell is credited with inventing the telephone in 1876."
        ],
        response="Alexander Graham Bell invented the telephone.",
        reference="Alexander Graham Bell"
    ),
]
dataset = EvaluationDataset(samples=samples)

# 3. Run evaluation
result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
        SemanticSimilarity(embeddings=evaluator_emb),
    ],
    show_progress=True,
)

print(result)
# {'faithfulness': 1.0, 'context_recall': 1.0, 'context_precision': 1.0,
#  'factual_correctness': 1.0, 'answer_relevancy': 0.97, 'semantic_similarity': 0.99}
```

---

## 6. LLM Configuration

Ragas v0.4 provides two approaches to configure the judge LLM.

### Approach 1: llm_factory() — native Ragas API

```python
from ragas.llms import llm_factory

# OpenAI
from openai import OpenAI
client = OpenAI()
llm = llm_factory("gpt-4o", client=client)

# Anthropic Claude
import os
from anthropic import Anthropic
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
llm = llm_factory("claude-3-5-sonnet-20241022", provider="anthropic", client=client)

# Google Gemini
import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm = llm_factory(
    "gemini-2.0-flash",
    provider="google",
    client=genai.GenerativeModel("gemini-2.0-flash")
)

# Local Ollama
from openai import OpenAI
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
llm = llm_factory("mistral", provider="openai", client=client)
```

### Approach 2: LangchainLLMWrapper — LangChain models

```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# OpenAI via LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# AWS Bedrock via LangChain
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
evaluator_llm = LangchainLLMWrapper(
    ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-east-1")
)
evaluator_emb = LangchainEmbeddingsWrapper(
    BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")
)

# Azure OpenAI via LangChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
evaluator_llm = LangchainLLMWrapper(
    AzureChatOpenAI(
        azure_endpoint="https://your-resource.openai.azure.com/",
        azure_deployment="gpt-4o",
        api_version="2024-02-01"
    )
)

# Google AI Studio via LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.0-flash"))
```

### Setting LLM globally on a metric

```python
from ragas.metrics.collections import Faithfulness
from ragas.run_config import RunConfig

metric = Faithfulness()
metric.llm = evaluator_llm
metric.init(RunConfig())  # initializes the metric with the given LLM
```

---

## 7. LangChain Integration

Ragas integrates with LangChain at two levels:

### Level 1 — Using LangChain LLMs/Embeddings as the judge

Use `LangchainLLMWrapper` and `LangchainEmbeddingsWrapper` (see Section 6, Approach 2).

> **Deprecation note:** In v0.4, these wrappers are soft-deprecated. The new preferred approach is to use `llm_factory()`. The wrappers remain functional but may be removed in a future major version.

### Level 2 — Evaluating a LangChain-built RAG pipeline

Full end-to-end workflow:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections import LLMContextRecall, Faithfulness, FactualCorrectness

# --- Build a LangChain RAG pipeline ---
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()

docs = [Document(page_content="Paris is the capital of France.")]
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "Answer using only the context:\nContext: {context}\nQuestion: {query}"
)
qa_chain = prompt | llm | StrOutputParser()

# --- Collect evaluation data ---
test_queries = ["What is the capital of France?"]
references   = ["Paris"]

raw_data = []
for query, ref in zip(test_queries, references):
    docs = retriever.invoke(query)
    response = qa_chain.invoke({
        "context": "\n".join(d.page_content for d in docs),
        "query": query
    })
    raw_data.append({
        "user_input": query,
        "retrieved_contexts": [d.page_content for d in docs],
        "response": response,
        "reference": ref,
    })

evaluation_dataset = EvaluationDataset.from_list(raw_data)

# --- Evaluate ---
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
    ],
    llm=evaluator_llm,
)
print(result)
```

---

## 8. Results — EvaluationResult

`evaluate()` returns an `EvaluationResult` object.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `list[dict[str, Any]]` | Per-sample metric scores |
| `dataset` | `EvaluationDataset` | The original dataset |
| `binary_columns` | `list[str]` | Which metrics returned binary scores |
| `cost_cb` | `CostCallbackHandler \| None` | Cost info (if token_usage_parser was set) |
| `traces` | `list[dict]` | Execution traces |
| `run_id` | `UUID \| None` | Unique run identifier |

### Accessing results

```python
result = evaluate(dataset, metrics=[...])

# Print aggregate scores
print(result)
# {'context_recall': 1.0, 'faithfulness': 0.857, 'factual_correctness': 0.728}

# Convert to pandas DataFrame (one row per sample)
df = result.to_pandas()
print(df.columns.tolist())
# ['user_input', 'retrieved_contexts', 'response', 'reference',
#  'context_recall', 'faithfulness', 'factual_correctness']

# Raw per-sample scores
for row in result.scores:
    print(row)
# {'context_recall': 1.0, 'faithfulness': 1.0, 'factual_correctness': 0.8}
# {'context_recall': 1.0, 'faithfulness': 0.7, 'factual_correctness': 0.66}

# Aggregate mean per metric
import pandas as pd
df = result.to_pandas()
print(df[["context_recall", "faithfulness", "factual_correctness"]].mean())
```

### Cost tracking (optional)

```python
from ragas.cost import get_token_usage_for_openai

result = evaluate(
    dataset=dataset,
    metrics=[...],
    token_usage_parser=get_token_usage_for_openai,
)

# After evaluation
total_cost = result.total_cost(
    cost_per_input_token=0.000005,    # $5 per million input tokens
    cost_per_output_token=0.000015,   # $15 per million output tokens
)
print(f"Evaluation cost: ${total_cost:.4f}")
```

---

## 9. Custom Metrics

### Option A: Decorator approach (v0.4+, recommended for simple cases)

```python
from ragas.metrics import discrete_metric, numeric_metric, ranking_metric
from ragas.metrics import MetricResult

# Discrete metric: returns one of a fixed set of labels
@discrete_metric(name="tone_check", allowed_values=["formal", "informal", "neutral"])
async def tone_check(llm, user_input: str, response: str) -> MetricResult:
    prompt = f"Classify the tone of this response as 'formal', 'informal', or 'neutral'.\nResponse: {response}"
    result = await llm.agenerate([prompt])
    tone = result.generations[0][0].text.strip().lower()
    return MetricResult(value=tone, reason=f"Detected tone: {tone}")


# Numeric metric: returns a float in a range
@numeric_metric(name="conciseness", min_value=0.0, max_value=1.0)
async def conciseness(llm, user_input: str, response: str) -> MetricResult:
    prompt = f"Rate the conciseness of this response from 0 (verbose) to 1 (very concise).\nResponse: {response}\nRespond with only a number."
    result = await llm.agenerate([prompt])
    score = float(result.generations[0][0].text.strip())
    return MetricResult(value=score)
```

### Option B: Class-based approach (for complex logic)

```python
from dataclasses import dataclass, field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricResult
from ragas import SingleTurnSample

@dataclass
class HallucinationScore(MetricWithLLM, SingleTurnMetric):
    """Inverts Faithfulness: measures how much the response hallucinates."""
    name: str = "hallucination_score"
    _required_columns: dict = field(
        default_factory=lambda: {
            "single_turn": {"user_input", "response", "retrieved_contexts"}
        }
    )

    def __post_init__(self):
        from ragas.metrics.collections import Faithfulness
        self._faithfulness = Faithfulness()

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks=None
    ) -> float:
        self._faithfulness.llm = self.llm
        faith_result = await self._faithfulness._single_turn_ascore(sample, callbacks)
        # faith_result is a MetricResult or float depending on version
        faith_score = faith_result.value if hasattr(faith_result, "value") else faith_result
        return 1.0 - faith_score


# Usage
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
custom_metric = HallucinationScore(llm=evaluator_llm)

result = evaluate(
    dataset=dataset,
    metrics=[custom_metric],
    llm=evaluator_llm,
)
```

### Option C: DiscreteMetric / SimpleLLMMetric (prompt-based, no subclassing)

```python
from ragas.metrics import DiscreteMetric
from openai import AsyncOpenAI
from ragas.llms import llm_factory

client = AsyncOpenAI()
evaluator_llm = llm_factory("gpt-4o", client=client)

metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="""Evaluate if this summary is accurate.
Response: {response}
Reference: {reference}
Answer with only 'accurate' or 'inaccurate'."""
)

import asyncio

async def main():
    score = await metric.ascore(
        llm=evaluator_llm,
        response="Einstein developed quantum mechanics.",
        reference="Einstein developed the theory of relativity."
    )
    print(score.value)   # "inaccurate"
    print(score.reason)  # explanation from the LLM

asyncio.run(main())
```

---

## 10. Key Imports Cheatsheet

```python
# ── Core ────────────────────────────────────────────────────────────────────
from ragas import evaluate
from ragas import EvaluationDataset
from ragas import SingleTurnSample
from ragas import MultiTurnSample

# ── Dataset helpers ──────────────────────────────────────────────────────────
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample, EvaluationDataset

# ── Messages (for MultiTurnSample) ──────────────────────────────────────────
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

# ── Metrics (canonical v0.4 path) ────────────────────────────────────────────
from ragas.metrics.collections import (
    Faithfulness,
    ResponseRelevancy,       # was: AnswerRelevancy
    ContextPrecision,        # was: ContextPrecision (LLM-based)
    ContextRecall,           # was: LLMContextRecall
    FactualCorrectness,      # was: AnswerCorrectness
    SemanticSimilarity,      # was: AnswerSimilarity
    NoiseSensitivity,
    ContextEntityRecall,
    AspectCritic,
    SimpleCriteriaScore,
    BleuScore,
    RougeScore,
    FaithfulnesswithHHEM,
)

# ── Metrics (legacy path — still works but deprecated) ───────────────────────
from ragas.metrics import (
    LLMContextRecall,        # -> use ContextRecall
    Faithfulness,
    FactualCorrectness,
)

# ── Custom metric decorators ─────────────────────────────────────────────────
from ragas.metrics import discrete_metric, numeric_metric, ranking_metric
from ragas.metrics import MetricResult

# ── Custom metric base classes ───────────────────────────────────────────────
from ragas.metrics.base import (
    Metric,
    MetricWithLLM,
    MetricWithEmbeddings,
    SingleTurnMetric,
    MultiTurnMetric,
    DiscreteMetric,
    SimpleLLMMetric,
)

# ── LLM & Embeddings (Ragas-native) ─────────────────────────────────────────
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings  # Ragas native

# ── LLM & Embeddings (LangChain wrappers) ────────────────────────────────────
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── LlamaIndex wrappers ──────────────────────────────────────────────────────
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

# ── Configuration ─────────────────────────────────────────────────────────────
from ragas.run_config import RunConfig

# ── Cost tracking ─────────────────────────────────────────────────────────────
from ragas.cost import get_token_usage_for_openai
```

---

## API Stability Notes (v0.3 → v0.4 Migration)

| Old name | New name | Import path |
|----------|----------|-------------|
| `AnswerRelevancy` | `ResponseRelevancy` | `ragas.metrics.collections` |
| `AnswerSimilarity` | `SemanticSimilarity` | `ragas.metrics.collections` |
| `AnswerCorrectness` | `FactualCorrectness` | `ragas.metrics.collections` |
| `LLMContextRecall` | `ContextRecall` | `ragas.metrics.collections` |
| `ground_truths` field | `reference` field | `SingleTurnSample` |
| `instructor_llm_factory()` | `llm_factory()` | `ragas.llms` |
| Dataclass-based prompts | Function-based prompts | internal |
| `evaluate()` (primary) | `@experiment()` decorator | `ragas` (experiment-centric) |

> `evaluate()` is still fully functional in v0.4 but the framework's preferred workflow for new projects is the `@experiment()` decorator pattern, which adds structured tracking, CSV logging, and reproducibility.

---

*Sources:*
- *https://docs.ragas.io/en/stable/*
- *https://docs.ragas.io/en/stable/references/evaluate/*
- *https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/*
- *https://docs.ragas.io/en/stable/concepts/components/eval_dataset/*
- *https://docs.ragas.io/en/latest/concepts/components/eval_sample/*
- *https://docs.ragas.io/en/stable/howtos/integrations/langchain/*
- *https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/*
- *https://docs.ragas.io/en/v0.1.21/references/metrics.html*
