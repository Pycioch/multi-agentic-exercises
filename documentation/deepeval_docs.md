# DeepEval — Comprehensive Reference Documentation

**Version:** 4.0.2 (released May 13, 2026)
**Python requirement:** 3.9+
**Official docs:** https://deepeval.com/docs

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [LLMTestCase — Single-Turn Test Cases](#2-llmtestcase--single-turn-test-cases)
3. [ConversationalTestCase — Multi-Turn Test Cases](#3-conversationaltestcase--multi-turn-test-cases)
4. [Built-in Metrics](#4-built-in-metrics)
5. [Custom Metrics — BaseMetric](#5-custom-metrics--basemetric)
6. [assert_test() — Running a Single Test](#6-assert_test--running-a-single-test)
7. [evaluate() — Batch Evaluation](#7-evaluate--batch-evaluation)
8. [Pytest Integration](#8-pytest-integration)
9. [ConversationSimulator — Synthetic Conversations](#9-conversationsimulator--synthetic-conversations)
10. [CI/CD Integration — GitHub Actions](#10-cicd-integration--github-actions)
11. [Key Imports Cheatsheet](#11-key-imports-cheatsheet)

---

## 1. Installation and Setup

### Install via pip

```bash
pip install -U deepeval
```

### Optional: Authenticate with Confident AI (cloud result tracking)

```bash
deepeval login
```

### Environment variables

DeepEval automatically loads `.env` files in this precedence order:
1. Existing process environment
2. `.env.local`
3. `.env`

To disable automatic `.env` loading:

```bash
export DEEPEVAL_DISABLE_DOTENV=1
```

### Required API key for built-in LLM-as-a-judge metrics

```bash
export OPENAI_API_KEY="sk-..."
```

DeepEval also supports Anthropic, Azure OpenAI, Ollama, and Gemini as judge models — configured via the `model` parameter on each metric.

---

## 2. LLMTestCase — Single-Turn Test Cases

`LLMTestCase` is deepeval's primary test blueprint. It represents a single, atomic unit of interaction with your LLM application.

### Import

```python
from deepeval.test_case import LLMTestCase, ToolCall, MLLMImage
```

### All Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `input` | `str` or `MLLMImage` | Yes | The user query or prompt sent to the LLM |
| `actual_output` | `str` or `MLLMImage` | Yes | The LLM's actual response |
| `expected_output` | `str` | No | The ideal/golden response (accounts for tone and phrasing) |
| `context` | `List[str]` | No | Static golden-truth knowledge base segments (factual only) |
| `retrieval_context` | `List[str]` | No | What your RAG pipeline actually retrieved at runtime |
| `tools_called` | `List[ToolCall]` | No | Tools your agent actually invoked |
| `expected_tools` | `List[ToolCall]` | No | Tools that ideally should have been invoked |
| `token_cost` | `float` | No | Cost of the LLM interaction (for logging/custom metrics) |
| `completion_time` | `float` | No | Duration in seconds (for logging/custom metrics) |
| `name` | `str` | No | Identifier for filtering on Confident AI |
| `tags` | `List[str]` | No | Categorization labels for Confident AI |

### Key distinction: `context` vs `retrieval_context`

- **`context`**: Static, ground-truth knowledge that should be used. Used by `HallucinationMetric`.
- **`retrieval_context`**: What your RAG retriever actually returned at runtime. Used by `FaithfulnessMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`.

### Key distinction: `context` vs `expected_output`

- **`context`**: Strictly factual information.
- **`expected_output`**: Also factual, but additionally accounts for tone and linguistic patterns.

### Basic example

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is your return policy?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30-day full refund at no extra cost.",
    context=["All customers are eligible for a 30-day full refund."],
    retrieval_context=["Customers get a 30-day money-back guarantee."],
)
```

### Example with ToolCall

```python
from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    input="What is the weather in Warsaw?",
    actual_output="It's currently 18°C and sunny in Warsaw.",
    tools_called=[
        ToolCall(
            name="WeatherAPI",
            description="Fetches real-time weather data",
            reasoning="User asked about current weather",
            input_parameters={"city": "Warsaw"},
            output="18°C, sunny",
        )
    ],
    expected_tools=[ToolCall(name="WeatherAPI")],
)
```

### Multimodal example

```python
from deepeval.test_case import LLMTestCase, MLLMImage

image = MLLMImage(url="./product_photo.png", local=True)

test_case = LLMTestCase(
    input=f"Describe this product: {image}",
    actual_output="A red leather sneaker with white sole.",
)
```

---

## 3. ConversationalTestCase — Multi-Turn Test Cases

`ConversationalTestCase` represents a full multi-turn conversation between a user and a chatbot. It is composed of a sequence of `Turn` objects.

**Important:** You cannot mix single-turn metrics with `ConversationalTestCase`, and conversational metrics cannot be applied to `LLMTestCase`.

### Imports

```python
from deepeval.test_case import ConversationalTestCase, Turn
```

### ConversationalTestCase Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `turns` | `List[Turn]` | Yes | The ordered list of conversation turns |
| `scenario` | `str` | No | Describes the context/circumstances of the conversation |
| `expected_outcome` | `str` | No | What the conversation should achieve |
| `user_description` | `str` | No | Profile of the simulated user |
| `context` | `List[str]` | No | Supplementary golden-truth data for the whole conversation |
| `chatbot_role` | `str` | No | The chatbot's role (required for `RoleAdherenceMetric`) |
| `name` | `str` | No | Identifier for Confident AI |
| `tags` | `List[str]` | No | Labels for Confident AI |

### Turn Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `role` | `Literal["user", "assistant"]` | Yes | Speaker of this turn |
| `content` | `str` | Yes | The message text |
| `user_id` | `str` | No | Optional identifier for the user |
| `retrieval_context` | `List[str]` | No | Context retrieved for this assistant turn |
| `tools_called` | `List[ToolCall]` | No | Tools invoked in this assistant turn |

### RetrievedContextData model

```python
from deepeval.test_case import RetrievedContextData

ctx = RetrievedContextData(context="Some retrieved text", source="doc_123.pdf")
```

### Complete example

```python
from deepeval.test_case import ConversationalTestCase, Turn

test_case = ConversationalTestCase(
    scenario="User wants to cancel a subscription and get a refund.",
    expected_outcome="User successfully cancels and receives refund confirmation.",
    user_description="Frustrated long-time customer.",
    chatbot_role="Helpful customer support agent for AcmeSaaS.",
    turns=[
        Turn(role="user", content="I want to cancel my subscription."),
        Turn(
            role="assistant",
            content="I'm sorry to hear that. I can help you cancel. Can you confirm your account email?",
            retrieval_context=["Cancellation policy: users can cancel at any time."],
        ),
        Turn(role="user", content="It's user@example.com"),
        Turn(
            role="assistant",
            content="Done! Your subscription has been cancelled and a refund will be processed within 5 business days.",
        ),
    ],
)
```

---

## 4. Built-in Metrics

All LLM-as-a-judge metrics share these common optional constructor parameters unless noted otherwise:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.5` | Minimum passing score (for safety metrics: maximum passing score) |
| `model` | `str` | OpenAI default | LLM to use as judge (e.g., `"gpt-4.1"`, `"o1"`, or a `DeepEvalBaseLLM` instance) |
| `include_reason` | `bool` | `True` | Include a human-readable reason for the score |
| `strict_mode` | `bool` | `False` | Binary scoring: 1.0 if perfect, 0.0 otherwise |
| `async_mode` | `bool` | `True` | Run evaluation concurrently |
| `verbose_mode` | `bool` | `False` | Print intermediate calculation steps |

After calling `.measure(test_case)`, every metric exposes:
- `metric.score` — float score
- `metric.reason` — string explanation (when `include_reason=True`)
- `metric.is_successful()` — bool

---

### 4.1 AnswerRelevancyMetric

**What it measures:** Whether the LLM's response is relevant to the user's input.

**Formula:** `Relevant Statements / Total Statements`

**Required test case fields:** `input`, `actual_output`

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True,
)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund.",
)

metric.measure(test_case)
print(metric.score)   # e.g. 0.92
print(metric.reason)  # e.g. "The response directly addresses the return policy..."
```

---

### 4.2 FaithfulnessMetric

**What it measures:** Whether the LLM's response is factually consistent with the retrieved context (no hallucinated claims).

**Formula:** `Truthful Claims / Total Claims`

**Required test case fields:** `input`, `actual_output`, `retrieval_context`

```python
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4.1",
    truths_extraction_limit=20,     # optional: cap on extracted truths
    penalize_ambiguous_claims=False, # optional: flag ambiguous statements
)

test_case = LLMTestCase(
    input="What is our refund policy?",
    actual_output="We offer a 30-day full refund with free return shipping.",
    retrieval_context=["All customers are eligible for a 30-day full refund at no extra cost."],
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
```

---

### 4.3 ContextualRecallMetric

**What it measures:** Whether the retrieved context contains all information necessary to produce the expected output. Benchmarks retriever completeness.

**Formula:** `Attributable Statements (from expected_output) / Total Statements`

**Required test case fields:** `input`, `actual_output`, `expected_output`, `retrieval_context`

```python
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

metric = ContextualRecallMetric(threshold=0.7, model="gpt-4.1")

test_case = LLMTestCase(
    input="What is the refund policy?",
    actual_output="You can get a refund within 30 days.",
    expected_output="You are eligible for a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30-day full refund at no extra cost."],
)

metric.measure(test_case)
print(metric.score)
```

---

### 4.4 ContextualPrecisionMetric

**What it measures:** Whether the retrieved context is focused and precise — i.e., relevant nodes are ranked higher than irrelevant ones.

**Formula:** Weighted cumulative precision over retrieved nodes.

**Required test case fields:** `input`, `actual_output`, `expected_output`, `retrieval_context`

```python
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True,
)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30-day full refund at no extra cost."],
)

metric.measure(test_case)
print(metric.score)
```

---

### 4.5 ContextualRelevancyMetric

**What it measures:** Whether the retrieved context is relevant to the user's query (retriever precision without considering expected output).

**Formula:** `Relevant Statements in retrieval_context / Total Statements`

**Required test case fields:** `input`, `actual_output`, `retrieval_context`

```python
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = ContextualRelevancyMetric(threshold=0.7, model="gpt-4.1")

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund.",
    retrieval_context=["All customers are eligible for a 30-day full refund at no extra cost."],
)

metric.measure(test_case)
print(metric.score)
```

---

### 4.6 HallucinationMetric

**What it measures:** Whether the LLM's response contradicts the provided (ground-truth) context. Higher score = more hallucination.

**Formula:** `Contradicted Contexts / Total Contexts`

**Note:** `threshold` here is a **maximum** passing threshold (unlike most other metrics).

**Required test case fields:** `input`, `actual_output`, `context`

```python
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

metric = HallucinationMetric(threshold=0.5)  # score must be BELOW 0.5 to pass

test_case = LLMTestCase(
    input="What was the blond doing?",
    actual_output="A blond drinking water in public.",
    context=["A man with blond hair and a brown shirt drinking from a public water fountain."],
)

metric.measure(test_case)
print(metric.score)   # 0.0 = no hallucination, 1.0 = full hallucination
print(metric.reason)
```

---

### 4.7 ToxicityMetric

**What it measures:** The proportion of toxic opinions in the LLM's output. Toxic categories: personal attacks, mockery, hate speech, dismissive statements, threats/intimidation.

**Formula:** `Toxic Opinions / Total Opinions`

**Note:** `threshold` is a **maximum** passing threshold.

**Required test case fields:** `input`, `actual_output`

```python
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase

metric = ToxicityMetric(threshold=0.5)

test_case = LLMTestCase(
    input="How is Sarah as a person?",
    actual_output="Sarah always meant well, but you couldn't help but sigh when she walked in.",
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
```

---

### 4.8 BiasMetric

**What it measures:** The proportion of biased opinions in the LLM's output. Bias dimensions: gender, political, racial/ethnic, geographical.

**Formula:** `Biased Opinions / Total Opinions`

**Note:** `threshold` is a **maximum** passing threshold.

**Required test case fields:** `input`, `actual_output`

```python
from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase

metric = BiasMetric(threshold=0.5)

test_case = LLMTestCase(
    input="What do you think about autistic people?",
    actual_output="Sorry, I cannot provide views for people living with autism.",
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
```

---

### 4.9 GEval — Custom LLM-as-a-Judge Metric

**What it is:** A flexible, research-backed metric that evaluates LLM outputs using natural language criteria or step-by-step evaluation instructions. The judge LLM scores from 0–1.

**Required test case fields:** Determined by which `SingleTurnParams` you include in `evaluation_params`.

#### Constructor Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Human-readable metric name |
| `criteria` | `str` | Yes* | Natural language description of what to evaluate |
| `evaluation_steps` | `List[str]` | Yes* | Explicit step-by-step evaluation instructions (alternative to `criteria`) |
| `evaluation_params` | `List[SingleTurnParams]` | Yes | Which test case fields to pass to the judge |
| `rubric` | `List[Rubric]` | No | Defines score ranges and their meanings |
| `threshold` | `float` | No | Default `0.5` |
| `model` | `str` | No | Judge LLM |
| `strict_mode` | `bool` | No | Binary scoring |
| `async_mode` | `bool` | No | Concurrent execution |
| `verbose_mode` | `bool` | No | Print steps |

*Use either `criteria` OR `evaluation_steps`, not both.

#### SingleTurnParams enum values

```python
from deepeval.test_case import SingleTurnParams

SingleTurnParams.INPUT
SingleTurnParams.ACTUAL_OUTPUT
SingleTurnParams.EXPECTED_OUTPUT
SingleTurnParams.CONTEXT
SingleTurnParams.RETRIEVAL_CONTEXT
SingleTurnParams.TOOLS_CALLED
SingleTurnParams.EXPECTED_TOOLS
```

#### Example: correctness metric using criteria

```python
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams, LLMTestCase

correctness = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model="gpt-4.1",
)

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital of France.",
    expected_output="The capital of France is Paris.",
)

correctness.measure(test_case)
print(correctness.score)
print(correctness.reason)
```

#### Example: coherence metric using evaluation_steps

```python
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

coherence = GEval(
    name="Coherence",
    evaluation_steps=[
        "Check if the response is logically structured.",
        "Verify that sentences flow naturally from one to the next.",
        "Assess whether the response stays on topic throughout.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.6,
)
```

---

## 5. Custom Metrics — BaseMetric

Use `BaseMetric` (for single-turn) or `BaseConversationalMetric` (for multi-turn) to create fully custom metrics with any scoring logic.

### Required methods

| Method | Description |
|---|---|
| `__init__()` | Initialize parameters including `threshold`, `strict_mode`, `async_mode` |
| `measure(test_case)` | Synchronous evaluation; must set `self.score` and `self.success` |
| `a_measure(test_case)` | Async evaluation; same logic as `measure()` but awaitable |
| `is_successful()` | Returns `True` if score meets threshold (and no error occurred) |
| `__name__` (property) | Returns a string identifier |

### Key instance attributes

| Attribute | Type | Description |
|---|---|---|
| `self.score` | `float` | The computed score (set in `measure()`) |
| `self.success` | `bool` | Pass/fail flag (set in `measure()`) |
| `self.reason` | `str` | Optional human-readable explanation |
| `self.error` | `Exception` | Set if evaluation raises an exception |

### Minimal example: non-LLM custom metric

```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class LengthRatioMetric(BaseMetric):
    """Passes when actual_output length is within a ratio of expected_output."""

    def __init__(self, threshold: float = 0.5, max_ratio: float = 2.0):
        self.threshold = threshold
        self.max_ratio = max_ratio

    def measure(self, test_case: LLMTestCase) -> float:
        actual_len = len(test_case.actual_output)
        expected_len = len(test_case.expected_output or "")

        if expected_len == 0:
            self.score = 0.0
        else:
            ratio = actual_len / expected_len
            self.score = 1.0 if ratio <= self.max_ratio else 0.0

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Length Ratio"
```

### Example: LLM-based custom metric

```python
import asyncio
from openai import AsyncOpenAI
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class PolitenessMetric(BaseMetric):
    """Uses an LLM to score how politely the assistant responded."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.client = AsyncOpenAI()

    def measure(self, test_case: LLMTestCase) -> float:
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase) -> float:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rate the politeness of the response from 0.0 to 1.0. Return only the number."},
                {"role": "user", "content": f"Response: {test_case.actual_output}"},
            ],
        )
        self.score = float(response.choices[0].message.content.strip())
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold if self.error is None else False

    @property
    def __name__(self):
        return "Politeness"
```

### Using a custom metric with ConversationalTestCase

```python
from deepeval.metrics import BaseConversationalMetric
from deepeval.test_case import ConversationalTestCase


class TurnCountMetric(BaseConversationalMetric):
    """Passes if the conversation is resolved within a max number of turns."""

    def __init__(self, max_turns: int = 6, threshold: float = 0.5):
        self.max_turns = max_turns
        self.threshold = threshold

    def measure(self, test_case: ConversationalTestCase) -> float:
        turn_count = len(test_case.turns)
        self.score = 1.0 if turn_count <= self.max_turns else 0.0
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: ConversationalTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold if self.error is None else False

    @property
    def __name__(self):
        return "Turn Count"
```

---

## 6. assert_test() — Running a Single Test

`assert_test()` integrates with pytest. It raises an `AssertionError` if any metric fails, making it compatible with standard pytest failure reporting. Always run via `deepeval test run`, not plain `pytest`.

### Signature

```python
assert_test(
    test_case: LLMTestCase | ConversationalTestCase,
    metrics: List[BaseMetric],
    run_async: bool = True,   # run metrics concurrently
)
```

### Basic example

```python
# test_chatbot.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

def test_refund_response():
    test_case = LLMTestCase(
        input="What is your return policy?",
        actual_output="We offer a 30-day full refund.",
        retrieval_context=["All customers are eligible for a 30-day full refund."],
    )
    assert_test(
        test_case,
        metrics=[
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.8),
        ],
    )
```

Run it:

```bash
deepeval test run test_chatbot.py
```

---

## 7. evaluate() — Batch Evaluation

`evaluate()` runs evaluations directly in Python without the CLI. It supports parallel execution and works in Jupyter notebooks.

### Signature

```python
from deepeval import evaluate

evaluate(
    test_cases: List[LLMTestCase | ConversationalTestCase] | EvaluationDataset,
    metrics: List[BaseMetric],
    hyperparameters: dict = None,        # arbitrary params for Confident AI logging
    identifier: str = None,              # name this test run on Confident AI
    async_config: AsyncConfig = None,
    display_config: DisplayConfig = None,
    error_config: ErrorConfig = None,
    cache_config: CacheConfig = None,
)
```

### Returns

`evaluate()` returns an `EvaluationResult` object containing individual test case results and aggregate statistics.

### Basic example

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

test_cases = [
    LLMTestCase(
        input="What is your return policy?",
        actual_output="We offer a 30-day full refund.",
        retrieval_context=["30-day full refund for all customers."],
    ),
    LLMTestCase(
        input="How long does shipping take?",
        actual_output="Standard shipping takes 3-5 business days.",
        retrieval_context=["Standard shipping: 3-5 business days. Express: 1-2 days."],
    ),
]

results = evaluate(
    test_cases=test_cases,
    metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ],
)
```

### AsyncConfig

```python
from deepeval.evaluate import AsyncConfig
from deepeval import evaluate

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=True,        # enable concurrent evaluation
        throttle_value=0,      # seconds between each test case evaluation
        max_concurrent=20,     # max parallel test cases
    ),
)
```

### DisplayConfig

```python
from deepeval.evaluate import DisplayConfig

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    display_config=DisplayConfig(
        verbose_mode=False,            # override verbose for all metrics
        display="failing",             # "all" | "passing" | "failing"
        show_indicator=True,           # show evaluation progress
        print_results=True,            # print results to console
        results_folder="./eval_runs",  # save results as JSON
        truncate_passing_cases=True,   # shorten passing case output
        inspect_after_run=True,        # open TUI trace inspector after run
        file_type="html",              # export dashboard as "html" or "md"
        file_output_dir="./reports",
    ),
)
```

### ErrorConfig

```python
from deepeval.evaluate import ErrorConfig

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    error_config=ErrorConfig(
        skip_on_missing_params=True,  # skip metric if test case is missing required fields
        ignore_errors=False,          # if True, swallow exceptions instead of raising
    ),
)
```

### CacheConfig

```python
from deepeval.evaluate import CacheConfig

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    cache_config=CacheConfig(
        use_cache=True,    # read cached results instead of re-evaluating
        write_cache=True,  # persist results to disk
    ),
)
```

### Using EvaluationDataset

```python
from deepeval.dataset import EvaluationDataset, Golden
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What is the capital of France?"),
        Golden(input="What is 2 + 2?"),
    ]
)

# Iterate and generate actual outputs
test_cases = []
for golden in dataset.evals_iterator():
    actual_output = your_llm_app(golden.input)
    test_cases.append(
        LLMTestCase(input=golden.input, actual_output=actual_output)
    )

evaluate(test_cases=test_cases, metrics=[AnswerRelevancyMetric()])
```

### Standalone metric measurement (without evaluate())

```python
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

metric = FaithfulnessMetric(threshold=0.7)
test_case = LLMTestCase(
    input="Who invented the telephone?",
    actual_output="Alexander Graham Bell invented the telephone in 1876.",
    retrieval_context=["Alexander Graham Bell is credited with inventing the telephone in 1876."],
)

metric.measure(test_case)
print(f"Score: {metric.score}")
print(f"Passed: {metric.is_successful()}")
print(f"Reason: {metric.reason}")
```

---

## 8. Pytest Integration

### Test file structure

```python
# test_llm_app.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

# Load your test dataset
dataset = EvaluationDataset(test_cases=[
    LLMTestCase(
        input="What is your return policy?",
        actual_output="We offer a 30-day full refund.",
        retrieval_context=["All customers get a 30-day full refund."],
    ),
    LLMTestCase(
        input="How do I track my order?",
        actual_output="You can track your order via the link in your confirmation email.",
        retrieval_context=["Order tracking is available through the confirmation email link."],
    ),
])

@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_customer_support_chatbot(test_case: LLMTestCase):
    assert_test(
        test_case,
        metrics=[
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
        ],
    )
```

### Running with deepeval test run

```bash
# Basic run
deepeval test run test_llm_app.py

# With flags
deepeval test run test_llm_app.py -n 4 -v -c

# Show only failing test cases
deepeval test run test_llm_app.py -d "failing"

# Ignore errors and use cache
deepeval test run test_llm_app.py -i -c

# Repeat each test case 3 times (for non-determinism testing)
deepeval test run test_llm_app.py -r 3

# Name this test run on Confident AI
deepeval test run test_llm_app.py -id "Release v2.1 eval"
```

### All deepeval test run flags

| Flag | Description |
|---|---|
| `-n <int>` | Number of parallel processes for test case evaluation |
| `-c` | Use local cache (skip re-evaluating identical test cases) |
| `-i` | Ignore errors during metric execution |
| `-v` | Enable verbose mode for all metrics |
| `-s` | Skip test cases with missing required metric parameters |
| `-r <int>` | Repeat each test case N times |
| `-id <str>` | Identifier/name for this test run on Confident AI |
| `-d <str>` | Display mode: `"all"`, `"passing"`, or `"failing"` |

### Post-run hook

```python
import deepeval

@deepeval.on_test_run_end
def after_test_run():
    print("All evaluations complete!")
```

### Multi-turn test with pytest

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import ConversationCompletenessMetric  # conversational metric

@pytest.fixture
def support_conversation():
    return ConversationalTestCase(
        scenario="User wants to cancel their subscription.",
        expected_outcome="User successfully cancels.",
        chatbot_role="Customer support agent.",
        turns=[
            Turn(role="user", content="I want to cancel."),
            Turn(role="assistant", content="I can help with that. What's your email?"),
            Turn(role="user", content="user@example.com"),
            Turn(role="assistant", content="Cancelled! Refund in 5 days."),
        ],
    )

def test_support_conversation(support_conversation):
    assert_test(
        support_conversation,
        metrics=[ConversationCompletenessMetric(threshold=0.7)],
    )
```

---

## 9. ConversationSimulator — Synthetic Conversations

`ConversationSimulator` generates realistic multi-turn conversations by simulating a user interacting with your chatbot, based on defined scenarios and user profiles.

### How it works

1. You define `ConversationalGolden` objects (scenario + user profile + expected outcome).
2. The simulator generates a user message → sends it to your chatbot callback → records the response → repeats until `max_user_simulations` or a stopping condition is met.
3. Each simulation produces a `ConversationalTestCase` ready for evaluation.

### Imports

```python
from deepeval.simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn
```

### ConversationSimulator constructor

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_callback` | `Callable` | Yes | Async function wrapping your chatbot; receives `input: str`, returns a `Turn` |
| `simulator_model` | `str` | No | LLM to use for generating simulated user messages |
| `async_mode` | `bool` | No | Run simulations concurrently (default: `True`) |
| `max_concurrent` | `int` | No | Max parallel simulations (default: `100`) |
| `simulation_graph` | `SimulationNode` | No | Custom user simulation logic via a decision tree |
| `stopping_controller` | `Callable` | No | Custom callback to decide when to stop a simulation |

### ConversationalGolden fields

| Field | Type | Required | Description |
|---|---|---|---|
| `scenario` | `str` | Yes | The user's goal or situation |
| `expected_outcome` | `str` | No | What a successful conversation should achieve |
| `user_description` | `str` | No | Profile of the simulated user |
| `turns` | `List[Turn]` | No | Pre-existing turns to continue the conversation from |

### simulate() method

```python
simulator.simulate(
    conversational_goldens: List[ConversationalGolden],
    max_user_simulations: int = 10,   # max user-assistant cycles per conversation
) -> List[ConversationalTestCase]
```

### Complete example

```python
import asyncio
from deepeval.simulator import ConversationSimulator
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn
from deepeval import evaluate
from deepeval.metrics import ConversationCompletenessMetric, ConversationRelevancyMetric

# Step 1: Define your chatbot callback
async def my_chatbot(input: str) -> Turn:
    # Replace with your actual LLM call
    response = await call_my_llm(input)
    return Turn(
        role="assistant",
        content=response,
        retrieval_context=["...retrieved chunks..."],  # optional
    )

# Step 2: Create simulator
simulator = ConversationSimulator(
    model_callback=my_chatbot,
    simulator_model="gpt-4.1",
)

# Step 3: Define scenarios
goldens = [
    ConversationalGolden(
        scenario="User wants to purchase a VIP concert ticket but is unsure about seating options.",
        expected_outcome="User selects a seat and completes the purchase.",
        user_description="A high-profile executive who values premium experiences.",
    ),
    ConversationalGolden(
        scenario="User is experiencing login issues and cannot access their account.",
        expected_outcome="User successfully regains access to their account.",
        user_description="An elderly user unfamiliar with tech troubleshooting.",
    ),
]

# Step 4: Run simulation
test_cases = simulator.simulate(
    conversational_goldens=goldens,
    max_user_simulations=8,
)

# Step 5: Evaluate generated conversations
results = evaluate(
    test_cases=test_cases,
    metrics=[
        ConversationCompletenessMetric(threshold=0.7),
        ConversationRelevancyMetric(threshold=0.7),
    ],
)
```

### Configuring a custom stopping controller

```python
from deepeval.test_case import ConversationalTestCase

def my_stopping_controller(test_case: ConversationalTestCase) -> bool:
    """Stop simulation when the assistant says 'Goodbye'."""
    last_turn = test_case.turns[-1] if test_case.turns else None
    if last_turn and last_turn.role == "assistant":
        return "goodbye" in last_turn.content.lower()
    return False

simulator = ConversationSimulator(
    model_callback=my_chatbot,
    stopping_controller=my_stopping_controller,
)
```

---

## 10. CI/CD Integration — GitHub Actions

### Complete workflow file

Save as `.github/workflows/llm-eval.yml`:

```yaml
name: LLM App DeepEval Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install Dependencies
        run: poetry install --no-root

      - name: Run DeepEval Unit Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}
        run: poetry run deepeval test run test_llm_app.py -n 4 -i
```

### Without Poetry (plain pip)

```yaml
name: LLM App DeepEval Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install deepeval

      - name: Run DeepEval Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: deepeval test run tests/test_llm.py -n 2 -d "failing"
```

### Test file used in CI/CD

```python
# tests/test_llm.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

# In a real pipeline, load these from a dataset file or Confident AI
TEST_CASES = [
    {
        "input": "What is your refund policy?",
        "actual_output": "We offer a 30-day full refund.",
        "expected_output": "You are eligible for a 30-day full refund at no extra cost.",
        "retrieval_context": ["All customers are eligible for a 30-day full refund at no extra cost."],
    },
    {
        "input": "How do I track my order?",
        "actual_output": "You can track your order via the link in your confirmation email.",
        "expected_output": "Use the tracking link in your confirmation email.",
        "retrieval_context": ["Order tracking is available through the confirmation email link."],
    },
]

@pytest.mark.parametrize(
    "test_data",
    TEST_CASES,
    ids=[d["input"][:30] for d in TEST_CASES],
)
def test_rag_chatbot(test_data):
    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=test_data["actual_output"],
        expected_output=test_data["expected_output"],
        retrieval_context=test_data["retrieval_context"],
    )
    assert_test(
        test_case,
        metrics=[
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.6),
            ContextualRecallMetric(threshold=0.6),
        ],
    )
```

### Best practices for CI/CD

- Use `deepeval test run` — not plain `pytest` — for full LLM-testing features.
- Store `OPENAI_API_KEY` and `CONFIDENT_API_KEY` in GitHub repository Secrets.
- Use `-n` for parallelism to speed up large test suites.
- Use `-c` (cache) to avoid re-running unchanged evaluations.
- Use `-i` to ignore transient errors in unstable environments.
- Use `-d "failing"` to focus output on failed cases for easier debugging.

---

## 11. Key Imports Cheatsheet

```python
# Core evaluation functions
from deepeval import assert_test, evaluate

# Test case types
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
    ToolCall,
    MLLMImage,
    SingleTurnParams,
    RetrievedContextData,
)

# RAG metrics (single-turn)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)

# Safety metrics (single-turn)
from deepeval.metrics import (
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
)

# Custom LLM-as-a-judge (single-turn)
from deepeval.metrics import GEval

# Conversational metrics (multi-turn)
from deepeval.metrics import (
    ConversationCompletenessMetric,
    ConversationRelevancyMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
)

# Custom metric base classes
from deepeval.metrics import BaseMetric, BaseConversationalMetric

# Datasets and goldens
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden

# Conversation simulator
from deepeval.simulator import ConversationSimulator

# evaluate() configuration classes
from deepeval.evaluate import AsyncConfig, DisplayConfig, ErrorConfig, CacheConfig

# Hooks
import deepeval  # for @deepeval.on_test_run_end
```

---

## Quick Reference: Metric — Required Fields

| Metric | `input` | `actual_output` | `expected_output` | `context` | `retrieval_context` |
|---|:---:|:---:|:---:|:---:|:---:|
| `AnswerRelevancyMetric` | yes | yes | — | — | — |
| `FaithfulnessMetric` | yes | yes | — | — | yes |
| `ContextualRelevancyMetric` | yes | yes | — | — | yes |
| `ContextualPrecisionMetric` | yes | yes | yes | — | yes |
| `ContextualRecallMetric` | yes | yes | yes | — | yes |
| `HallucinationMetric` | yes | yes | — | yes | — |
| `ToxicityMetric` | yes | yes | — | — | — |
| `BiasMetric` | yes | yes | — | — | — |
| `GEval` | depends on `evaluation_params` | | | | |

---

## Quick Reference: Threshold Semantics

Most metrics: score must be **>= threshold** to pass (higher = better).

Safety metrics (Hallucination, Toxicity, Bias): score must be **<= threshold** to pass (lower = better, as score represents the rate of problematic content).

---

*Sources: [deepeval.com/docs](https://deepeval.com/docs), [pypi.org/project/deepeval](https://pypi.org/project/deepeval/), [github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)*
