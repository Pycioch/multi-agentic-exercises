# LangChain Reference Documentation

> Covers: `langchain==1.3.1` · `langchain-openai==1.2.1` · `langchain-core==1.4.0`
> Python ≥ 3.10 required. All three packages are the **latest stable** releases as of May 2026.

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [ChatOpenAI](#2-chatopenai)
3. [create_agent()](#3-create_agent)
4. [langchain_core.messages](#4-langchain_coremessages)
5. [langchain_core.tools](#5-langchain_coretools)
6. [CallbackHandlers](#6-callbackhandlers)
7. [Memory and Conversation History](#7-memory-and-conversation-history)
8. [LCEL — LangChain Expression Language](#8-lcel--langchain-expression-language)
9. [Key Imports Cheatsheet](#9-key-imports-cheatsheet)

---

## 1. Installation and Setup

### Install packages

```bash
# Core trio
pip install langchain==1.3.1 langchain-openai==1.2.1 langchain-core==1.4.0

# Convenience shorthand (installs langchain + openai extras)
pip install "langchain[openai]"

# For agent persistence (checkpointing)
pip install langgraph
```

### Environment variables

```bash
export OPENAI_API_KEY="sk-..."          # required for ChatOpenAI
export OPENAI_ORG_ID="org-..."          # optional
export LANGSMITH_TRACING=true           # optional — enables LangSmith tracing
export LANGSMITH_API_KEY="ls__..."      # required when tracing is on
```

Or set programmatically before instantiating the model:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Verify installation

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

---

## 2. ChatOpenAI

**Import:** `from langchain_openai import ChatOpenAI`

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | — | Model name, e.g. `"gpt-4o"`, `"gpt-4o-mini"`, `"o3"` |
| `temperature` | `float` | `0.7` | Sampling temperature (0–2). `None` uses model default |
| `max_tokens` | `int \| None` | `None` | Max tokens to generate (deprecated alias for `max_completion_tokens`) |
| `max_completion_tokens` | `int \| None` | `None` | Max tokens for completion |
| `api_key` | `str \| None` | `None` | Falls back to `OPENAI_API_KEY` env var |
| `base_url` | `str \| None` | `None` | Override API base URL (for proxies, local servers) |
| `organization` | `str \| None` | `None` | Falls back to `OPENAI_ORG_ID` env var |
| `timeout` | `float \| tuple[float,float] \| None` | `None` | Request timeout in seconds |
| `max_retries` | `int \| None` | `2` | Number of retry attempts on failure |
| `stream_options` | `dict` | `{}` | E.g. `{"include_usage": True}` for token counts during streaming |
| `logprobs` | `bool \| None` | `None` | Return log-probabilities of tokens |
| `model_kwargs` | `dict` | `{}` | Extra kwargs passed directly to `openai.chat.completions.create()` |
| `extra_body` | `dict \| None` | `None` | Provider-specific parameters |
| `use_responses_api` | `bool \| None` | `None` | Use OpenAI Responses API (stateful conversations) |
| `reasoning` | `dict \| None` | `None` | Reasoning config, e.g. `{"effort": "high"}` for o-series models |

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=1024,
    timeout=30,
    max_retries=3,
    api_key="sk-...",          # or via OPENAI_API_KEY env var
)
```

---

### `invoke(input, config=None, **kwargs) -> AIMessage`

Sends a single request and returns a complete `AIMessage`.

**Input formats accepted:**

```python
# 1. List of message objects
from langchain_core.messages import HumanMessage, SystemMessage

response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
])
print(response.content)          # "Paris"
print(response.usage_metadata)   # {"input_tokens": 28, "output_tokens": 5, ...}

# 2. List of (role, content) tuples — shorthand
response = llm.invoke([
    ("system", "You are helpful."),
    ("human", "Explain LangChain in one sentence."),
])

# 3. Plain string (wrapped as HumanMessage internally)
response = llm.invoke("What year is it?")
```

**Accessing the result:**

```python
response.content          # str — the text response
response.text             # str — same as content (convenience alias)
response.tool_calls       # list[dict] — populated when the model called tools
response.usage_metadata   # dict — token counts
response.response_metadata  # dict — finish_reason, model, etc.
response.id               # str — unique message id
```

---

### `stream(input, config=None, **kwargs) -> Iterator[AIMessageChunk]`

Yields incremental chunks. Each chunk is an `AIMessageChunk` with a `.content` string fragment. Chunks can be summed with `+` to reconstruct a full `AIMessage`.

```python
# Basic streaming
for chunk in llm.stream("Tell me a joke about Python."):
    print(chunk.content, end="", flush=True)

# With token usage during streaming (requires stream_options)
llm_streaming = ChatOpenAI(
    model="gpt-4o",
    stream_options={"include_usage": True},
)
full = None
for chunk in llm_streaming.stream("Hello"):
    full = chunk if full is None else full + chunk
    print(chunk.content, end="", flush=True)

print(full.usage_metadata)  # available on the accumulated message
```

**Async streaming:**

```python
async def main():
    async for chunk in llm.astream("Tell me a story"):
        print(chunk.content, end="", flush=True)
```

---

### `bind_tools(tools, *, tool_choice=None, strict=False, **kwargs) -> ChatOpenAI`

Returns a new `ChatOpenAI` instance with tool schemas bound. The model can then emit `tool_calls` in its response.

**Accepted tool formats:**
- Python functions decorated with `@tool`
- Pydantic `BaseModel` subclasses
- `StructuredTool` instances
- Raw JSON-schema dicts

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Option A: @tool decorated function
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny, 22°C in {location}"

# Option B: Pydantic schema
class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, description="Max number of results")

# Bind tools to the model
llm_with_tools = llm.bind_tools(
    [get_weather, SearchInput],
    tool_choice="auto",  # "auto" | "required" | "none" | {"type": "function", "function": {"name": "..."}}
    strict=True,         # enforce JSON schema compliance
)

# Invoke — model may choose to call a tool
response = llm_with_tools.invoke("What's the weather in Warsaw?")
print(response.tool_calls)
# [{"name": "get_weather", "args": {"location": "Warsaw"}, "id": "call_abc123", "type": "tool_call"}]
```

**Processing tool calls manually:**

```python
from langchain_core.messages import ToolMessage

if response.tool_calls:
    for tc in response.tool_calls:
        result = get_weather.invoke(tc["args"])
        tool_msg = ToolMessage(
            content=result,
            tool_call_id=tc["id"],
            name=tc["name"],
        )
```

---

### `with_structured_output(schema, *, method="json_schema", include_raw=False)`

Constrains the model to always return data matching the schema. Returns a `Runnable` that outputs a Pydantic object (or dict).

```python
from pydantic import BaseModel

class Joke(BaseModel):
    setup: str
    punchline: str
    rating: int = Field(ge=1, le=10)

structured_llm = llm.with_structured_output(Joke)
joke = structured_llm.invoke("Tell me a programming joke")
print(joke.setup)      # type: str
print(joke.rating)     # type: int
```

---

### Async variants

| Sync | Async |
|------|-------|
| `invoke()` | `ainvoke()` |
| `stream()` | `astream()` |
| `batch()` | `abatch()` |

---

## 3. `create_agent()`

**Import:** `from langchain.agents import create_agent`

Builds a LangGraph-powered agent with a model-call → tool-call loop. Returns a `CompiledStateGraph` (not a plain chain) — it has `.invoke()`, `.stream()`, `.batch()` like any runnable.

### Full signature

```python
create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | type | dict | None = None,
    state_schema: type[AgentState] | None = None,
    context_schema: type | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    transformers: Sequence[Callable] | None = None,
) -> CompiledStateGraph
```

### Parameters explained

| Parameter | Description |
|-----------|-------------|
| `model` | A model string like `"openai:gpt-4o"` or an initialized `BaseChatModel` instance |
| `tools` | List of tools available to the agent. If `None` or empty, the agent is a plain model node (no tool loop) |
| `system_prompt` | Prepended as `SystemMessage` at the start of every model call |
| `middleware` | Ordered list of `AgentMiddleware` instances for custom logic hooks |
| `response_format` | Force structured output — pass a Pydantic class or schema |
| `state_schema` | Custom `TypedDict` extending `AgentState` for extra state fields |
| `context_schema` | Schema for runtime context passed at invocation time |
| `checkpointer` | Enables conversation persistence (e.g. `MemorySaver`, `SqliteSaver`) |
| `store` | External store for feature flags or user preferences |
| `interrupt_before` / `interrupt_after` | Node names where execution pauses (for human-in-the-loop) |
| `debug` | Print detailed node execution info |
| `name` | Name this agent when embedding as a subgraph |
| `cache` | `BaseCache` instance to cache graph execution results |

### Basic usage

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def check_weather(location: str) -> str:
    """Return the weather forecast for the specified location."""
    return f"It's always sunny in {location}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    return str(eval(expression))

# Option A: model string (LangChain resolves it)
agent = create_agent(
    model="openai:gpt-4o",
    tools=[check_weather, calculate],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

# Option B: pre-configured model instance
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm,
    tools=[check_weather, calculate],
    system_prompt="You are a helpful assistant.",
)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Kraków?"}]
})
print(result["messages"][-1].content)
```

### With memory (checkpointer)

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_agent(
    model="openai:gpt-4o",
    tools=[check_weather],
    checkpointer=memory,
)

config = {"configurable": {"thread_id": "user-session-42"}}

# Turn 1
agent.invoke({"messages": [("user", "My name is Alice")]}, config=config)

# Turn 2 — agent remembers "Alice" from the same thread
result = agent.invoke({"messages": [("user", "What's my name?")]}, config=config)
print(result["messages"][-1].content)  # "Your name is Alice."
```

### Streaming agent output

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Calculate 42 * 17"}]},
    stream_mode="updates",
):
    print(chunk)
```

---

## 4. `langchain_core.messages`

**Import:** `from langchain_core.messages import ...`

All message objects are Pydantic models and serializable.

---

### `BaseMessage`

The abstract parent of all message types. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str \| list[str \| dict]` | The message content (text or multimodal blocks) |
| `id` | `str \| None` | Unique identifier (auto-generated or provider-assigned) |
| `name` | `str \| None` | Optional name for the speaker/role |
| `additional_kwargs` | `dict` | Provider-specific extras (e.g. function_call from older APIs) |
| `response_metadata` | `dict` | Populated on AI responses (finish_reason, model, etc.) |

---

### `HumanMessage`

Represents input from the user.

```python
from langchain_core.messages import HumanMessage

# Simple text
msg = HumanMessage(content="What is the capital of Poland?")

# With name (for multi-user scenarios)
msg = HumanMessage(content="Hello!", name="alice", id="msg-001")

# Multimodal — text + image
msg = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
])
```

---

### `AIMessage`

Represents the model's response. Additional fields beyond `BaseMessage`:

| Field | Type | Description |
|-------|------|-------------|
| `tool_calls` | `list[ToolCall]` | Structured tool calls emitted by the model |
| `invalid_tool_calls` | `list[InvalidToolCall]` | Malformed tool calls (for error handling) |
| `usage_metadata` | `dict \| None` | Token counts: `input_tokens`, `output_tokens`, `total_tokens` |
| `text` | `str` | Convenience property — same as `content` when content is a string |
| `content_blocks` | `list[ContentBlock]` | Standardized content blocks (text, reasoning, image, etc.) |

```python
from langchain_core.messages import AIMessage

# Manually constructing (rare — usually returned by the model)
msg = AIMessage(content="Paris is the capital of France.")
print(msg.content)   # "Paris is the capital of France."
print(msg.text)      # same

# Tool call structure (from model response)
# msg.tool_calls -> [
#   {"name": "get_weather", "args": {"location": "Paris"}, "id": "call_abc", "type": "tool_call"}
# ]
```

---

### `SystemMessage`

Primes the model's behavior. Usually the first message in a conversation.

```python
from langchain_core.messages import SystemMessage

msg = SystemMessage(content="You are a senior Python developer. Reply in Polish.")

# Equivalent shorthand in invoke/create_agent:
# ("system", "You are a senior Python developer.")
```

---

### `ToolMessage`

Passes the result of tool execution back to the model. **Required fields:** `content`, `tool_call_id`.

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The tool's return value as a string |
| `tool_call_id` | `str` | Must match the `id` in the corresponding `AIMessage.tool_calls` entry |
| `name` | `str \| None` | Tool name (informational) |
| `artifact` | `Any` | Supplementary data NOT sent to the model (e.g. raw objects) |

```python
from langchain_core.messages import ToolMessage

tool_result = ToolMessage(
    content="22°C, sunny",
    tool_call_id="call_abc123",  # must match AIMessage.tool_calls[i]["id"]
    name="get_weather",
)
```

---

### `AIMessageChunk`

Returned by `.stream()`. Chunks accumulate incrementally and can be concatenated with `+`:

```python
from langchain_core.messages import AIMessageChunk

full_message = None
for chunk in llm.stream("Tell me a story"):
    if full_message is None:
        full_message = chunk
    else:
        full_message = full_message + chunk

print(type(full_message))  # <class 'langchain_core.messages.ai.AIMessage'>
print(full_message.content)
```

---

### Utility functions

```python
from langchain_core.messages import (
    convert_to_messages,       # convert dicts/tuples to message objects
    filter_messages,           # filter by type or id
    merge_message_runs,        # merge consecutive same-role messages
    trim_messages,             # trim to fit token budget
)

# Convert role/content tuples to message objects
messages = convert_to_messages([
    ("system", "You are helpful."),
    ("human", "Hi!"),
    ("ai", "Hello! How can I help?"),
    ("human", "What's 2+2?"),
])

# Trim to a token limit (useful before sending to model)
trimmed = trim_messages(
    messages,
    max_tokens=200,
    strategy="last",        # keep the last N tokens
    token_counter=llm,      # use the model's tokenizer
    include_system=True,    # always keep the SystemMessage
    start_on="human",       # first message after trimming must be human
)
```

---

## 5. `langchain_core.tools`

**Import:** `from langchain_core.tools import tool, StructuredTool, BaseTool`

---

### `@tool` decorator

The simplest way to create a tool. LangChain infers:
- **name** from function name (underscores → spaces)
- **description** from docstring (required — the model uses it to decide when to call)
- **args_schema** from type hints (required for correct schema generation)

```python
from langchain_core.tools import tool

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for up-to-date information.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
    """
    # implementation
    return f"Found {max_results} results for: {query}"

# Inspect
print(search_web.name)         # "search_web"
print(search_web.description)  # docstring text
print(search_web.args)         # {"query": {"type": "string"}, ...}

# Call directly (bypasses LLM)
result = search_web.invoke({"query": "LangChain tutorial", "max_results": 3})
```

**Custom name and description:**

```python
@tool("web_search", description="Use this to search the internet for current events.")
def search(query: str) -> str:
    return f"Results for: {query}"
```

**Async tools:**

```python
@tool
async def fetch_page(url: str) -> str:
    """Fetch the content of a web page."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
```

**Pydantic schema for complex inputs:**

```python
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    table: str = Field(description="Name of the database table")
    filters: dict = Field(default_factory=dict, description="WHERE clause filters as key-value pairs")
    limit: int = Field(default=10, ge=1, le=1000, description="Max rows to return")

@tool(args_schema=DatabaseQueryInput)
def query_database(table: str, filters: dict, limit: int) -> str:
    """Query a database table with optional filters."""
    return f"Queried {table} with {filters}, limit={limit}"
```

---

### `StructuredTool`

For creating tools from existing callables without the decorator, or when you need more control.

```python
from langchain_core.tools import StructuredTool

def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

# From a function
add_tool = StructuredTool.from_function(
    func=add_numbers,
    name="add",                         # optional, defaults to function name
    description="Add two numbers",      # optional, defaults to docstring
)

result = add_tool.invoke({"a": 3.5, "b": 2.0})  # 5.5

# From function + async function
async def add_async(a: float, b: float) -> float:
    return a + b

add_tool = StructuredTool.from_function(
    func=add_numbers,
    coroutine=add_async,
    name="add",
)
```

**Direct construction:**

```python
from pydantic import BaseModel

class MultiplyInput(BaseModel):
    x: int = Field(description="First factor")
    y: int = Field(description="Second factor")

def multiply(x: int, y: int) -> int:
    return x * y

multiply_tool = StructuredTool(
    name="multiply",
    description="Multiply two integers.",
    args_schema=MultiplyInput,
    func=multiply,
)
```

---

### `BaseTool` (subclassing)

Use when you need instance state, custom error handling, or complex initialization.

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

class MySearchTool(BaseTool):
    name: str = "my_search"
    description: str = "Search internal knowledge base for information."
    args_schema: Type[BaseModel] = SearchInput

    # Instance state (set in __init__ or as class vars)
    api_key: str = ""
    base_url: str = "https://search.example.com"

    def _run(self, query: str) -> str:
        """Synchronous implementation — must be implemented."""
        # call your API here
        return f"Results for '{query}' from {self.base_url}"

    async def _arun(self, query: str) -> str:
        """Async implementation — optional, defaults to running _run in thread."""
        return self._run(query)

tool = MySearchTool(api_key="secret-123")
result = tool.invoke({"query": "LangChain agents"})
```

**BaseTool key fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Exposed to the model |
| `description` | `str` | required | When/how to use this tool |
| `args_schema` | `type[BaseModel] \| None` | `None` | Input validation schema |
| `return_direct` | `bool` | `False` | If `True`, agent returns tool output directly (stops loop) |
| `handle_tool_error` | `bool \| str \| Callable` | `False` | Error recovery strategy |
| `handle_validation_error` | `bool \| str \| Callable` | `False` | Validation failure recovery |
| `response_format` | `"content" \| "content_and_artifact"` | `"content"` | Return type mapping |

**Error handling:**

```python
class SafeTool(BaseTool):
    name: str = "safe_tool"
    description: str = "A tool with error handling"
    handle_tool_error: bool = True          # returns generic error message on exception
    # handle_tool_error: str = "Tool failed, try again"    # custom message
    # handle_tool_error: Callable = lambda e: f"Error: {e}"  # dynamic message

    def _run(self, input: str) -> str:
        if not input:
            raise ToolException("Input cannot be empty")
        return f"Processed: {input}"
```

---

## 6. CallbackHandlers

Callbacks hook into LangChain's event system at every stage: model calls, chain execution, tool invocations.

**Import:** `from langchain_core.callbacks import BaseCallbackHandler`

### `BaseCallbackHandler` — key event methods

| Method | When called | Key parameters |
|--------|-------------|----------------|
| `on_llm_start(serialized, prompts, *, run_id, ...)` | Before non-chat LLM call | `prompts: list[str]` |
| `on_chat_model_start(serialized, messages, *, run_id, ...)` | Before chat model call | `messages: list[list[BaseMessage]]` |
| `on_llm_end(response, *, run_id, ...)` | After LLM/chat model completes | `response: LLMResult` |
| `on_llm_error(error, *, run_id, ...)` | On LLM/chat model error | `error: Exception` |
| `on_chain_start(serialized, inputs, *, run_id, ...)` | Before chain/runnable starts | `inputs: dict` |
| `on_chain_end(outputs, *, run_id, ...)` | After chain/runnable completes | `outputs: dict` |
| `on_chain_error(error, *, run_id, ...)` | On chain error | `error: Exception` |
| `on_tool_start(serialized, input_str, *, run_id, ...)` | Before tool execution | `input_str: str` |
| `on_tool_end(output, *, run_id, ...)` | After tool execution | `output: str` |
| `on_tool_error(error, *, run_id, ...)` | On tool error | `error: Exception` |

### Implementing a custom handler

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from typing import Any
from uuid import UUID

class LoggingCallbackHandler(BaseCallbackHandler):
    """Logs all LangChain events to console."""

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model_name = serialized.get("kwargs", {}).get("model_name", "unknown")
        print(f"[LLM START] model={model_name}, run_id={run_id}")
        for msg_list in messages:
            for msg in msg_list:
                print(f"  [{msg.__class__.__name__}] {msg.content[:80]}")

    def on_llm_end(self, response, *, run_id: UUID, **kwargs: Any) -> None:
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        print(f"[LLM END] tokens={usage}, run_id={run_id}")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        print(f"[TOOL START] {tool_name}({input_str})")

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        print(f"[TOOL END] output={output[:80]}")

    def on_chain_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        print(f"[CHAIN ERROR] {type(error).__name__}: {error}")
```

### Passing callbacks — the config dict pattern

Callbacks are passed at runtime via the `config` parameter (a `RunnableConfig` dict). They apply to the invoked runnable **and all its children** (model calls, tool calls, sub-chains).

```python
from langchain_core.runnables import RunnableConfig

handler = LoggingCallbackHandler()

# Pattern 1: plain dict (most common)
response = llm.invoke(
    [HumanMessage(content="Hello")],
    config={"callbacks": [handler]},
)

# Pattern 2: RunnableConfig typed dict
config: RunnableConfig = {
    "callbacks": [handler],
    "tags": ["production", "user-facing"],
    "metadata": {"user_id": "u-123", "session": "abc"},
    "run_name": "ChatSession",
    "max_concurrency": 4,
    "recursion_limit": 25,
    "configurable": {"model": "gpt-4o"},
}
response = llm.invoke([HumanMessage(content="Hello")], config=config)

# Pattern 3: pass at construction time (applies to every call)
llm_with_callbacks = ChatOpenAI(
    model="gpt-4o",
    callbacks=[handler],   # constructor-level callbacks
)
```

### `RunnableConfig` fields reference

| Field | Type | Description |
|-------|------|-------------|
| `callbacks` | `list[BaseCallbackHandler]` | Event handlers for this run and all sub-runs |
| `tags` | `list[str]` | Labels attached to all runs in this call |
| `metadata` | `dict[str, Any]` | Arbitrary JSON-serializable key-value data |
| `run_name` | `str` | Label for the root run in traces |
| `run_id` | `UUID \| None` | Force a specific run ID |
| `max_concurrency` | `int \| None` | Max parallel operations in `batch()` |
| `recursion_limit` | `int` | Max recursion depth (default: 25) |
| `configurable` | `dict[str, Any]` | Dynamic config values for `RunnableConfigurableFields` |

---

## 7. Memory and Conversation History

LangChain 1.x uses two main patterns for memory. The modern preferred approach uses `create_agent` with a **checkpointer**. The LCEL approach uses `RunnableWithMessageHistory`.

---

### Pattern A: Agent with checkpointer (recommended for agents)

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()  # in-memory; use SqliteSaver for persistence

agent = create_agent(
    model="openai:gpt-4o",
    tools=[my_tool],
    checkpointer=memory,
)

# Each unique thread_id is a separate conversation
config = {"configurable": {"thread_id": "conversation-1"}}

# Turn 1
agent.invoke({"messages": [("user", "My name is Alice and I like cats.")]}, config=config)

# Turn 2 — agent has full history from thread "conversation-1"
result = agent.invoke({"messages": [("user", "What do you know about me?")]}, config=config)
print(result["messages"][-1].content)
# "You told me your name is Alice and that you like cats."
```

**Persistent checkpointer (SQLite):**

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("conversations.db", check_same_thread=False)
memory = SqliteSaver(conn)

agent = create_agent(model="openai:gpt-4o", tools=[], checkpointer=memory)
```

---

### Pattern B: `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`

For LCEL chains (not agents).

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),   # injected conversation history
    ("human", "{input}"),
])

chain = prompt | llm

# In-memory store: session_id -> InMemoryChatMessageHistory
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",       # key in the chain input dict
    history_messages_key="history",   # must match MessagesPlaceholder variable_name
)

config = {"configurable": {"session_id": "user-alice"}}

response = chain_with_memory.invoke({"input": "Hi! My name is Alice."}, config=config)
print(response.content)

response = chain_with_memory.invoke({"input": "What's my name?"}, config=config)
print(response.content)  # "Your name is Alice."
```

---

### Pattern C: Manual history management

Build the message list yourself and pass it to `.invoke()`:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

history: list = [
    SystemMessage(content="You are a helpful assistant."),
]

def chat(user_input: str) -> str:
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(response)  # AIMessage goes back into history
    return response.content

print(chat("What's 2 + 2?"))
print(chat("Now multiply that by 3."))  # model remembers "4" from previous turn
```

---

### Trimming long histories

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    history,
    max_tokens=2000,
    strategy="last",        # "last" keeps most recent; "first" keeps oldest
    token_counter=llm,
    include_system=True,    # always keep the first SystemMessage
    start_on="human",       # ensure history starts with a human turn
    allow_partial=False,
)
```

---

## 8. LCEL — LangChain Expression Language

LCEL is the standard way to compose LangChain components into chains. The `|` operator chains runnables sequentially: output of the left side becomes input of the right side.

All LCEL components implement the `Runnable` protocol: `.invoke()`, `.stream()`, `.batch()`, `.ainvoke()`, `.astream()`, `.abatch()`.

---

### Basic chain composition

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Python expert."),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

# Chain: prompt -> llm -> parser
chain = prompt | llm | parser

# invoke returns a str (after StrOutputParser)
result = chain.invoke({"question": "What is a decorator in Python?"})
print(result)  # plain string answer

# stream works end-to-end
for token in chain.stream({"question": "Explain generators"}):
    print(token, end="", flush=True)

# batch runs multiple inputs in parallel
results = chain.batch([
    {"question": "What is __init__?"},
    {"question": "What is __repr__?"},
])
```

---

### `RunnablePassthrough`

Passes the input unchanged. Used to preserve the original input value alongside transformed values.

```python
from langchain_core.runnables import RunnablePassthrough

# Simple passthrough
passthrough = RunnablePassthrough()
print(passthrough.invoke("hello"))  # "hello"

# Common use: preserve the question in a RAG chain
from langchain_core.runnables import RunnableParallel

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,   # retriever returns docs, format_docs converts to string
        question=RunnablePassthrough(),    # passes original question through unchanged
    )
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is LCEL?")
```

---

### `RunnableParallel`

Runs multiple runnables simultaneously and merges results into a dict.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# Parallel execution — both sides run at the same time
parallel = RunnableParallel(
    joke=ChatPromptTemplate.from_template("Tell a joke about {topic}") | llm | StrOutputParser(),
    poem=ChatPromptTemplate.from_template("Write a haiku about {topic}") | llm | StrOutputParser(),
)

result = parallel.invoke({"topic": "Python"})
print(result["joke"])   # a joke
print(result["poem"])   # a haiku
```

---

### `RunnableLambda`

Wraps a plain Python function to make it usable in a chain.

```python
from langchain_core.runnables import RunnableLambda

def uppercase(text: str) -> str:
    return text.upper()

chain = llm | StrOutputParser() | RunnableLambda(uppercase)
result = chain.invoke([HumanMessage(content="Say hello")])
print(result)  # "HELLO!"

# Shorthand — Python functions are automatically coerced when used with |
# so RunnableLambda wrapper is often optional at the end of a chain
chain = llm | StrOutputParser() | (lambda x: x.upper())
```

---

### `RunnableBranch`

Conditional routing based on input.

```python
from langchain_core.runnables import RunnableBranch

math_chain = ChatPromptTemplate.from_template("Solve this math problem: {input}") | llm
general_chain = ChatPromptTemplate.from_template("Answer this question: {input}") | llm

branch = RunnableBranch(
    (lambda x: "math" in x["input"].lower(), math_chain),   # condition, runnable
    (lambda x: "code" in x["input"].lower(), code_chain),
    general_chain,  # default (fallback)
)

chain = {"input": RunnablePassthrough()} | branch
result = chain.invoke({"input": "What is 42 * 17?"})
```

---

### Prompt templates

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)

# ChatPromptTemplate — for chat models
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    MessagesPlaceholder("history"),   # inject list of messages
    ("human", "{input}"),
])

# Fill in variables
messages = prompt.invoke({
    "role": "Python tutor",
    "history": [],
    "input": "What is a list comprehension?",
})

# PromptTemplate — for simple string prompts
simple_prompt = PromptTemplate.from_template(
    "Translate the following to {language}: {text}"
)
```

---

### Output parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,       # extracts content as plain string
    JsonOutputParser,      # parses JSON from response
    PydanticOutputParser,  # parses into a Pydantic model
)

# StrOutputParser — most common
chain = prompt | llm | StrOutputParser()

# JsonOutputParser
from langchain_core.output_parsers import JsonOutputParser

json_chain = prompt | llm | JsonOutputParser()
result = json_chain.invoke({"input": "Return a JSON with name and age"})
print(result)  # dict

# PydanticOutputParser
from langchain_core.output_parsers import PydanticOutputParser

class PersonInfo(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=PersonInfo)
# Add format instructions to prompt
format_instructions = parser.get_format_instructions()
chain = prompt | llm | parser
```

---

### Async chains

All LCEL chains are automatically async-compatible:

```python
import asyncio

async def main():
    result = await chain.ainvoke({"question": "What is async in Python?"})
    print(result)

    async for token in chain.astream({"question": "Explain coroutines"}):
        print(token, end="", flush=True)

    results = await chain.abatch([
        {"question": "Q1"},
        {"question": "Q2"},
        {"question": "Q3"},
    ])

asyncio.run(main())
```

---

## 9. Key Imports Cheatsheet

```python
# ── Core model ──────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI

# ── Messages ────────────────────────────────────────────────────────────────
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    ToolMessage,
    AnyMessage,             # Union type: HumanMessage | AIMessage | ...
    convert_to_messages,
    filter_messages,
    trim_messages,
    merge_message_runs,
)

# ── Tools ────────────────────────────────────────────────────────────────────
from langchain_core.tools import tool, StructuredTool, BaseTool
from langchain_core.tools import ToolException, InjectedToolCallId

# ── Agents ──────────────────────────────────────────────────────────────────
from langchain.agents import create_agent

# ── LCEL / Runnables ─────────────────────────────────────────────────────────
from langchain_core.runnables import (
    Runnable,
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
    RunnableConfig,
    RunnableWithFallbacks,
)

# ── Prompts ──────────────────────────────────────────────────────────────────
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ── Output parsers ───────────────────────────────────────────────────────────
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)

# ── Callbacks ────────────────────────────────────────────────────────────────
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager

# ── Chat history / Memory ─────────────────────────────────────────────────────
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── LangGraph (for agents with memory) ───────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver  # requires langgraph-checkpoint-sqlite

# ── Type aliases ─────────────────────────────────────────────────────────────
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
```

---

## Quick reference — common patterns

```python
# 1. Minimal chat
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")
print(llm.invoke("Hello!").content)

# 2. Simple tool-calling agent
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

agent = create_agent(model="openai:gpt-4o", tools=[greet])
result = agent.invoke({"messages": [("user", "Greet Alice")]})

# 3. LCEL chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)
print(chain.invoke({"question": "What is Python?"}))

# 4. Streaming with callbacks
from langchain_core.callbacks import BaseCallbackHandler

class PrintHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        print(f"\n[tokens used: {response.llm_output}]")

for chunk in chain.stream(
    {"question": "Explain decorators"},
    config={"callbacks": [PrintHandler()]},
):
    print(chunk, end="", flush=True)

# 5. Structured output
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    points: list[str]

structured = ChatOpenAI(model="gpt-4o").with_structured_output(Summary)
result = structured.invoke("Summarize the benefits of type hints in Python")
print(result.title)
print(result.points)
```

---

*Sources: [langchain PyPI](https://pypi.org/project/langchain/) · [langchain-openai PyPI](https://pypi.org/project/langchain-openai/) · [langchain-core PyPI](https://pypi.org/project/langchain-core/) · [LangChain Docs](https://docs.langchain.com/oss/python/langchain/overview) · [ChatOpenAI Integration Docs](https://docs.langchain.com/oss/python/integrations/chat/openai) · [create_agent Reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent) · [LangChain Messages Docs](https://docs.langchain.com/oss/python/langchain/messages) · [LangChain Tools Docs](https://docs.langchain.com/oss/python/langchain/tools) · [RunnableConfig Reference](https://reference.langchain.com/python/langchain-core/runnables/langchain_core.runnables.config.RunnableConfig) · [LCEL Overview](https://www.aurelio.ai/learn/langchain-lcel)*
