# LangGraph Comprehensive Reference
**Version:** 1.2.0 (latest stable, May 2026)  
**Install:** `pip install -U langgraph`  
**Python:** 3.10+

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [StateGraph — Core Graph API](#2-stategraph--core-graph-api)
3. [State Definition Patterns](#3-state-definition-patterns)
4. [Node Functions](#4-node-functions)
5. [Conditional Edges and Routing](#5-conditional-edges-and-routing)
6. [Checkpointers](#6-checkpointers)
7. [Human-in-the-Loop](#7-human-in-the-loop)
8. [Subgraphs](#8-subgraphs)
9. [Supervisor Pattern](#9-supervisor-pattern)
10. [Swarm / Network Pattern](#10-swarm--network-pattern)
11. [Streaming](#11-streaming)
12. [Key Imports Cheatsheet](#12-key-imports-cheatsheet)

---

## 1. Installation and Setup

### Core package

```bash
pip install -U langgraph
```

### Optional extras

```bash
# Postgres checkpointer for production
pip install langgraph-checkpoint-postgres

# Pre-built supervisor multi-agent library
pip install langgraph-supervisor

# Pre-built swarm multi-agent library
pip install langgraph-swarm

# LangChain integrations (LLMs, tools)
pip install langchain-openai langchain-anthropic langchain-core
```

### Minimal working example

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    message: str

def hello_node(state: State) -> State:
    return {"message": "Hello, " + state["message"]}

builder = StateGraph(State)
builder.add_node(hello_node)
builder.add_edge(START, "hello_node")
builder.add_edge("hello_node", END)
graph = builder.compile()

result = graph.invoke({"message": "world"})
print(result)  # {'message': 'Hello, world'}
```

---

## 2. StateGraph — Core Graph API

### Constructor

```python
StateGraph(
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `state_schema` | **Required.** TypedDict or dataclass defining the graph state. |
| `context_schema` | Immutable runtime context injected at `invoke()` time (read-only in nodes). |
| `input_schema` | Narrows what the caller must provide to `invoke()`. Defaults to `state_schema`. |
| `output_schema` | Narrows what `invoke()` returns. Defaults to `state_schema`. |

### add_node()

```python
builder.add_node(
    node: str | Callable,
    action: Callable | None = None,
    *,
    defer: bool = False,
    metadata: dict | None = None,
    input_schema: type | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    error_handler: Callable | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
    timeout: float | timedelta | None = None,
) -> Self
```

| Parameter | Description |
|-----------|-------------|
| `node` | Node name (str) or a callable. If callable, the function name becomes the node name. |
| `action` | Node function when `node` is a string. |
| `defer` | Run this node at the very end of the step (after all other nodes). |
| `retry_policy` | Automatic retry on failure. |
| `cache_policy` | Cache node output by input hash. |
| `error_handler` | Fallback callable if the node raises. |
| `destinations` | Declare valid `Command(goto=...)` targets (required when nodes return `Command`). |
| `timeout` | Max seconds; async only. |

```python
# Automatic naming from function name
builder.add_node(my_node)           # registered as "my_node"

# Explicit name
builder.add_node("process", my_node)

# Fluent chaining
builder.add_node(node_a).add_node(node_b)
```

### add_edge()

```python
builder.add_edge(
    start_key: str | list[str],
    end_key: str,
) -> Self
```

- Single `start_key`: `end_key` runs after `start_key` finishes.
- List of `start_key`s: `end_key` waits for **all** listed nodes to finish (fan-in / join).

```python
builder.add_edge(START, "fetch_data")
builder.add_edge("fetch_data", "process")
builder.add_edge("process", END)

# Fan-in: "merge" runs only after both "a" and "b" finish
builder.add_edge(["a", "b"], "merge")
```

### add_conditional_edges()

```python
builder.add_conditional_edges(
    source: str,
    path: Callable[..., str | Sequence[str]],
    path_map: dict[str, str] | list[str] | None = None,
) -> Self
```

| Parameter | Description |
|-----------|-------------|
| `source` | Node whose completion triggers routing. |
| `path` | Callable receiving current state, returning next node name(s) or `"END"`. |
| `path_map` | Optional mapping from `path` return values to node names. Omit if `path` returns actual node names directly. |

```python
from typing import Literal
from langgraph.graph import END

def route(state: State) -> Literal["agent", "tools", "__end__"]:
    if state["next"] == "tools":
        return "tools"
    if state["next"] == "done":
        return "__end__"   # or END constant — same thing
    return "agent"

builder.add_conditional_edges("agent", route)

# With path_map (router returns simple keys, map resolves to node names)
def route_simple(state: State) -> str:
    return "a" if state["flag"] else "b"

builder.add_conditional_edges(
    "router",
    route_simple,
    {"a": "node_alpha", "b": "node_beta"},
)
```

**Tip:** Annotate router return type with `Literal[...]` so LangGraph can render accurate graph diagrams.

### compile()

```python
graph = builder.compile(
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
) -> CompiledStateGraph
```

| Parameter | Description |
|-----------|-------------|
| `checkpointer` | Persistence backend. Required for interrupts and multi-turn memory. |
| `interrupt_before` | Pause **before** executing these nodes (static breakpoints). |
| `interrupt_after` | Pause **after** executing these nodes (static breakpoints). |
| `debug` | Emit verbose execution logs. |

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"],
)
```

---

## 3. State Definition Patterns

### Plain TypedDict

```python
from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    result: str
    step_count: int
```

Each node returns a dict with the keys it wants to update. Keys not returned are left unchanged.

### Annotated fields with reducers

When multiple nodes write to the same key, LangGraph needs a **reducer** to merge the values. Use `Annotated[type, reducer]`.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    # append-only list: each node's returned list is concatenated
    messages: Annotated[list, operator.add]

    # last-write-wins (default when no annotation)
    status: str
```

#### Built-in reducers

| Reducer | Behaviour |
|---------|-----------|
| `operator.add` | Concatenates lists (or adds numbers) |
| `langgraph.graph.add_messages` | Appends messages, deduplicates by `id`, applies `RemoveMessage` deletions |

### MessagesState — the standard base

`MessagesState` is a pre-built TypedDict with a single `messages` key using `add_messages` as reducer. It is the standard starting point for chat agents.

```python
from langgraph.graph import MessagesState

# Equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

class MyState(MessagesState):
    # add custom fields alongside messages
    current_agent: str
    notes: Annotated[list[str], operator.add]
```

### Pydantic models as state

```python
from pydantic import BaseModel

class State(BaseModel):
    query: str = ""
    results: list[str] = []

builder = StateGraph(State)
```

---

## 4. Node Functions

### Signature

A node is any Python callable with one of these signatures:

```python
# Minimal — state in, partial state out
def my_node(state: State) -> dict:
    return {"key": "value"}

# With LangChain config (access callbacks, metadata, run_name)
from langchain_core.runnables import RunnableConfig

def my_node(state: State, config: RunnableConfig) -> dict:
    thread_id = config["configurable"].get("thread_id")
    return {"key": thread_id}
```

### Reading state

```python
def process_node(state: State) -> dict:
    query = state["query"]        # read
    history = state["messages"]   # read list
    return {"result": query.upper()}
```

### Writing state

Return a dict with only the keys you want to update. Missing keys stay unchanged.

```python
def update_node(state: State) -> dict:
    return {
        "status": "done",
        "messages": [AIMessage(content="Finished!")],  # appended by add_messages
    }
```

### Node returning Command (for dynamic routing)

```python
from langgraph.types import Command
from typing import Literal

def router_node(state: State) -> Command[Literal["path_a", "path_b"]]:
    if state["flag"]:
        return Command(goto="path_a", update={"status": "a"})
    return Command(goto="path_b", update={"status": "b"})
```

### Async nodes

```python
async def async_node(state: State) -> dict:
    result = await some_async_call(state["query"])
    return {"result": result}
```

---

## 5. Conditional Edges and Routing

### Router function (pure routing, no state update)

```python
from typing import Literal
from langgraph.graph import END

def should_continue(state: State) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"

builder.add_conditional_edges("agent", should_continue)
```

### END sentinel

`END` is a string constant `"__end__"`. Both forms are equivalent:

```python
from langgraph.graph import END

# These are identical:
return "__end__"
return END
```

### Router with path_map

```python
def route_by_type(state: State) -> str:
    return state["next_step"]  # returns "bill", "tech", or "done"

builder.add_conditional_edges(
    "supervisor",
    route_by_type,
    {
        "bill": "billing_agent",
        "tech": "tech_agent",
        "done": END,
    },
)
```

### Fan-out: route to multiple nodes

```python
def fan_out(state: State) -> list[str]:
    return ["node_a", "node_b"]  # both run in parallel

builder.add_conditional_edges("dispatcher", fan_out)
```

---

## 6. Checkpointers

Checkpointers persist graph state between steps, enabling multi-turn conversations, human-in-the-loop interrupts, and fault recovery.

### MemorySaver (in-memory, dev/test)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# thread_id isolates separate conversations
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [...]}, config=config)

# Inspect saved state
snapshot = graph.get_state(config)
print(snapshot.values)
print(snapshot.next)  # next nodes to run (empty if finished)
```

**Warning:** MemorySaver lives in RAM. A process restart wipes all state. Use only for development and tests.

### InMemorySaver (alias)

```python
from langgraph.checkpoint.memory import InMemorySaver  # same as MemorySaver
```

### AsyncPostgresSaver (production)

Install: `pip install langgraph-checkpoint-postgres`

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/mydb"

async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # Run this once to create tables and apply migrations
        await checkpointer.setup()

        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "session-abc"}}
        result = await graph.ainvoke({"messages": [...]}, config=config)
```

**Notes:**
- `setup()` is idempotent — safe to call on every startup.
- Requires `autocommit=True` if creating the connection manually.
- Default serializer: `JsonPlusSerializer` (handles LangChain types, datetimes, enums).
- For security, set env var `LANGGRAPH_STRICT_MSGPACK=true` in production.

### Synchronous PostgresSaver

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

### get_state() and update_state()

```python
# Read current checkpoint
snapshot = graph.get_state(config)
print(snapshot.values)      # current state dict
print(snapshot.next)        # list of next nodes
print(snapshot.created_at)  # timestamp

# Overwrite state values before resuming
graph.update_state(
    config,
    {"messages": [HumanMessage(content="corrected input")]},
)

# Resume
graph.invoke(None, config=config)
```

---

## 7. Human-in-the-Loop

### interrupt() — dynamic, inside a node

The preferred modern approach. Call `interrupt(value)` anywhere inside a node to pause execution. The value is surfaced to the caller. Execution resumes when `Command(resume=...)` is passed.

```python
from langgraph.types import interrupt, Command

def approval_node(state: State):
    # Pause and surface a question to the caller
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["pending_action"],
    })
    # When resumed, `decision` holds whatever was passed to Command(resume=...)
    return {"approved": decision}
```

**Full example:**

```python
from typing_extensions import TypedDict
from typing import Optional, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class State(TypedDict):
    task: str
    approved: Optional[bool]
    status: str

def human_review(state: State) -> Command[Literal["proceed", "cancel"]]:
    approved = interrupt(f"Review task: '{state['task']}'. Approve? (True/False)")
    if approved:
        return Command(goto="proceed", update={"approved": True})
    return Command(goto="cancel", update={"approved": False})

def proceed(state: State):
    return {"status": "completed"}

def cancel(state: State):
    return {"status": "cancelled"}

builder = StateGraph(State)
builder.add_node("human_review", human_review)
builder.add_node("proceed", proceed)
builder.add_node("cancel", cancel)
builder.add_edge(START, "human_review")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "run-1"}}

# First invocation — pauses at interrupt()
result = graph.invoke({"task": "Send email to client", "status": "pending"}, config, version="v2")
print(result.interrupts)
# > (Interrupt(value="Review task: 'Send email to client'. Approve? (True/False)"),)

# Human decides → resume
final = graph.invoke(Command(resume=True), config, version="v2")
print(final.value["status"])  # "completed"
```

### Validating input in a loop

`interrupt()` can be called multiple times in a while loop to re-ask until valid input arrives:

```python
def validated_input(state: State):
    prompt = "Enter a positive integer:"
    while True:
        value = interrupt(prompt)
        if isinstance(value, int) and value > 0:
            return {"count": value}
        prompt = f"'{value}' is invalid. Enter a positive integer:"
```

### interrupt() inside a tool

```python
from langchain_core.tools import tool

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email — requires human approval."""
    response = interrupt({
        "message": "Approve sending this email?",
        "to": to, "subject": subject, "body": body,
    })
    if response.get("action") == "approve":
        # optionally let human edit fields
        to = response.get("to", to)
        return f"Email sent to {to}"
    return "Email cancelled"
```

### Static breakpoints — interrupt_before / interrupt_after

Configured at compile time or passed to `invoke()`. Graph pauses automatically without any code changes in nodes.

```python
# Compile-time breakpoints
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["risky_operation"],
    interrupt_after=["generate_draft"],
)

config = {"configurable": {"thread_id": "t1"}}

# Run until breakpoint
graph.invoke(inputs, config)

# Inspect state, optionally edit it
snapshot = graph.get_state(config)
graph.update_state(config, {"draft": "manually corrected draft"})

# Resume (pass None as input)
graph.invoke(None, config)
```

```python
# Runtime breakpoints (override at invocation time)
graph.invoke(
    inputs,
    config=config,
    interrupt_before=["node_a"],
    interrupt_after=["node_b"],
)
```

### Handling interrupts in streaming (v2 API)

```python
async for chunk in graph.astream(initial_input, config, stream_mode=["updates"], version="v2"):
    if chunk["type"] == "updates" and "__interrupt__" in chunk["data"]:
        interrupt_value = chunk["data"]["__interrupt__"][0].value
        user_answer = await get_human_input(interrupt_value)
        # Restart loop with resume command
        async for chunk in graph.astream(Command(resume=user_answer), config, version="v2"):
            ...
        break
```

---

## 8. Subgraphs

A compiled `StateGraph` can be embedded as a node inside another graph, enabling modular, team-parallel development.

### Shared state schema (direct embedding)

When parent and subgraph share at least some state keys, pass the compiled subgraph directly to `add_node()`. Only overlapping keys flow back to the parent.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Subgraph
class SubgraphState(TypedDict):
    foo: str   # shared with parent
    bar: str   # private to subgraph — never leaks out

def sub_node_1(state: SubgraphState):
    return {"bar": "intermediate"}

def sub_node_2(state: SubgraphState):
    return {"foo": state["foo"] + "_" + state["bar"]}

sub_builder = StateGraph(SubgraphState)
sub_builder.add_node(sub_node_1)
sub_builder.add_node(sub_node_2)
sub_builder.add_edge(START, "sub_node_1")
sub_builder.add_edge("sub_node_1", "sub_node_2")
sub_builder.add_edge("sub_node_2", END)
subgraph = sub_builder.compile()

# Parent graph
class ParentState(TypedDict):
    foo: str

def parent_node_1(state: ParentState):
    return {"foo": "hello"}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("node_1", parent_node_1)
parent_builder.add_node("node_2", subgraph)   # <-- compiled subgraph as node
parent_builder.add_edge(START, "node_1")
parent_builder.add_edge("node_1", "node_2")
parent_builder.add_edge("node_2", END)
graph = parent_builder.compile()

print(graph.invoke({"foo": ""}))
# {'foo': 'hello_intermediate'}
```

### Different state schemas (node wrapper)

When schemas don't overlap, wrap the subgraph call in a regular node and translate manually.

```python
class SubState(TypedDict):
    bar: str   # completely different from parent

def subgraph_node_1(state: SubState):
    return {"bar": "hi! " + state["bar"]}

sub_builder = StateGraph(SubState)
sub_builder.add_node(subgraph_node_1)
sub_builder.add_edge(START, "subgraph_node_1")
sub_builder.add_edge("subgraph_node_1", END)
subgraph = sub_builder.compile()

class ParentState(TypedDict):
    foo: str

def call_subgraph(state: ParentState):
    # Translate parent → subgraph input
    output = subgraph.invoke({"bar": state["foo"]})
    # Translate subgraph output → parent state
    return {"foo": output["bar"]}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("call_subgraph", call_subgraph)
parent_builder.add_edge(START, "call_subgraph")
parent_builder.add_edge("call_subgraph", END)
graph = parent_builder.compile()
```

### Checkpointing in subgraphs

```python
# Default: subgraph inherits parent's checkpointer (recommended)
subgraph = sub_builder.compile()

# Give subgraph its own persistent memory (per-thread history)
subgraph = sub_builder.compile(checkpointer=True)

# Completely disable checkpointing for this subgraph
subgraph = sub_builder.compile(checkpointer=False)
```

### Streaming subgraph events

```python
for chunk in graph.stream(
    inputs,
    stream_mode="updates",
    subgraphs=True,    # include events from nested subgraphs
    version="v2",
):
    print(chunk["ns"])   # ('node_name:<task_id>',) for subgraph events
    print(chunk["data"])
```

---

## 9. Supervisor Pattern

A central **supervisor** node uses an LLM to route between specialist agents. Each specialist runs and returns control to the supervisor, which decides the next step.

```
START → supervisor → agent_A → supervisor → agent_B → supervisor → END
```

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END

# --- State ---
class SupervisorState(MessagesState):
    current_agent: str
    notes: Annotated[list[str], operator.add]

# --- Routing schema ---
class RoutingDecision(BaseModel):
    next_agent: Literal["billing", "tech_support", "account", "DONE"] = Field(
        description="Which agent handles next, or DONE if finished."
    )

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
routing_llm = llm.with_structured_output(RoutingDecision)

# --- Supervisor node ---
def supervisor(state: SupervisorState) -> dict:
    response = routing_llm.invoke([
        SystemMessage(content=(
            "You are a customer service supervisor. Choose which specialist "
            "should handle the next part of the request.\n"
            "Agents: billing, tech_support, account, DONE (when finished)."
        )),
        *state["messages"],
    ])
    return {"current_agent": response.next_agent}

# --- Routing function ---
def route_to_agent(state: SupervisorState) -> str:
    agent = state.get("current_agent", "DONE")
    return "end" if agent == "DONE" else agent

# --- Specialist nodes (each calls its own sub-agent or LLM) ---
def billing_node(state: SupervisorState) -> dict:
    # ... call billing specialist ...
    return {
        "messages": [AIMessage(content="Billing handled.")],
        "notes": ["Billing: processed refund"],
    }

def tech_node(state: SupervisorState) -> dict:
    return {
        "messages": [AIMessage(content="Tech issue resolved.")],
        "notes": ["Tech: fixed SSO certificate"],
    }

def account_node(state: SupervisorState) -> dict:
    return {
        "messages": [AIMessage(content="Account updated.")],
        "notes": ["Account: upgraded plan"],
    }

# --- Graph assembly ---
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("billing", billing_node)
builder.add_node("tech_support", tech_node)
builder.add_node("account", account_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "billing": "billing",
        "tech_support": "tech_support",
        "account": "account",
        "end": END,
    },
)
# All specialists return to supervisor
builder.add_edge("billing", "supervisor")
builder.add_edge("tech_support", "supervisor")
builder.add_edge("account", "supervisor")

supervisor_graph = builder.compile()

# --- Invocation ---
result = supervisor_graph.invoke({
    "messages": [HumanMessage(content="Fix my SSO and waive the setup fee.")],
    "current_agent": "",
    "notes": [],
})
```

### Using the langgraph-supervisor library

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

billing_agent = create_react_agent(llm, tools=[...], name="billing_agent")
tech_agent    = create_react_agent(llm, tools=[...], name="tech_agent")

supervisor = create_supervisor(
    agents=[billing_agent, tech_agent],
    model=llm,
    prompt="Route between billing and tech agents.",
)
graph = supervisor.compile()
result = graph.invoke({"messages": [HumanMessage(content="...")]})
```

---

## 10. Swarm / Network Pattern

Agents hand off directly to each other using `Command(goto=...)`. No central supervisor — each agent decides who handles the next step.

```
START → triage_agent
          ↕ (handoff tools)
        billing_agent ↔ tech_agent ↔ account_agent
```

### Command object

```python
from langgraph.types import Command

# Navigate to another node and update state
Command(
    goto="target_node",          # node name to jump to
    update={"key": "value"},     # optional state updates
    graph=Command.PARENT,        # use when inside a subgraph, to navigate the parent
)
```

### Manual handoff tool factory

```python
from langchain_core.tools import tool
from langgraph.types import Command

def make_handoff_tool(target_agent: str, description: str):
    @tool(f"transfer_to_{target_agent}")
    def handoff(reason: str) -> Command:
        """Transfer conversation to a specialist agent."""
        return Command(
            goto=target_agent,
            update={"current_agent": target_agent},
            graph=Command.PARENT,
        )
    handoff.__doc__ = description
    return handoff

transfer_to_billing = make_handoff_tool("billing", "Transfer to billing for invoices.")
transfer_to_tech    = make_handoff_tool("tech_support", "Transfer to tech for SSO issues.")
```

### Swarm agent nodes

```python
from langgraph.prebuilt import create_react_agent

triage = create_react_agent(
    llm,
    tools=[transfer_to_billing, transfer_to_tech],
    name="triage",
    prompt="Triage requests and hand off to the right specialist.",
)

billing = create_react_agent(
    llm,
    tools=[lookup_invoice, apply_discount, transfer_to_tech],
    name="billing",
    prompt="Handle billing questions. Delegate non-billing issues.",
)

tech = create_react_agent(
    llm,
    tools=[diagnose_sso, transfer_to_billing],
    name="tech_support",
    prompt="Handle technical issues. Delegate non-tech issues.",
)
```

### Swarm graph assembly

```python
from langgraph.graph import MessagesState, StateGraph, START, END

builder = StateGraph(MessagesState)
builder.add_node("triage", triage)
builder.add_node("billing", billing)
builder.add_node("tech_support", tech)

builder.add_edge(START, "triage")

# Each agent can re-enter itself or jump to another via Command
# No conditional edges needed when agents return Command(goto=...)

swarm_graph = builder.compile()
```

### Using the langgraph-swarm library

```python
from langgraph_swarm import create_swarm, create_handoff_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

billing_agent = create_react_agent(
    model=llm,
    tools=[lookup_invoice, create_handoff_tool(agent_name="tech_agent")],
    name="billing_agent",
    prompt="Billing specialist.",
)
tech_agent = create_react_agent(
    model=llm,
    tools=[diagnose_sso, create_handoff_tool(agent_name="billing_agent")],
    name="tech_agent",
    prompt="Tech specialist.",
)

swarm = create_swarm(
    agents=[billing_agent, tech_agent],
    default_active_agent="billing_agent",
)
graph = swarm.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "I need help with my invoice"}]},
    config,
)
```

### Supervisor vs. Swarm — when to use what

| | Supervisor | Swarm |
|--|------------|-------|
| Routing | Central LLM decides | Each agent decides |
| LLM calls per turn | +1 for supervisor | Fewer (direct handoffs) |
| Debuggability | High — single routing point | Lower — distributed |
| Best for | Predictable routing, early dev | Speed-critical, autonomous agents |

---

## 11. Streaming

### stream() and astream()

```python
# Synchronous
for chunk in graph.stream(
    input: dict | Command,
    config: dict | None = None,
    stream_mode: str | list[str] = "values",
    version: str = "v1",      # "v2" for unified format
    subgraphs: bool = False,
):
    ...

# Asynchronous
async for chunk in graph.astream(...):
    ...
```

### Stream modes

| Mode | What you get |
|------|-------------|
| `"values"` | Full state snapshot after each node. |
| `"updates"` | Only the state keys changed by each node. |
| `"messages"` | 2-tuples `(token_chunk, metadata)` emitted token-by-token from LLM calls. |
| `"custom"` | Arbitrary data emitted by nodes via `get_stream_writer()`. |
| `"checkpoints"` | Checkpoint events (requires checkpointer). |
| `"tasks"` | Task start/finish events with results and errors. |
| `"debug"` | Everything: checkpoints + tasks + extra metadata. |

### version="v2" unified format

All chunks follow a consistent dict:

```python
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),            # namespace tuple; non-empty for subgraph events
    "data": ...          # payload varies by type
}
```

### values mode

```python
for chunk in graph.stream({"query": "hello"}, stream_mode="values", version="v2"):
    if chunk["type"] == "values":
        print("Full state:", chunk["data"])
```

### updates mode

```python
for chunk in graph.stream({"query": "hello"}, stream_mode="updates", version="v2"):
    if chunk["type"] == "updates":
        for node_name, changes in chunk["data"].items():
            print(f"{node_name} updated:", changes)
```

### messages mode (token streaming)

```python
from langchain_core.messages import AIMessageChunk

for chunk in graph.stream({"messages": [...]}, stream_mode="messages", version="v2"):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk) and token.content:
            print(token.content, end="", flush=True)
        # metadata["langgraph_node"] tells you which node emitted this
```

### custom mode

```python
from langgraph.config import get_stream_writer

def my_node(state: State) -> dict:
    writer = get_stream_writer()
    writer({"progress": 25, "step": "fetching data"})
    # ... do work ...
    writer({"progress": 100, "step": "done"})
    return {"result": "finished"}

for chunk in graph.stream(inputs, stream_mode="custom", version="v2"):
    if chunk["type"] == "custom":
        print(chunk["data"])
        # {"progress": 25, "step": "fetching data"}
```

### Multiple modes simultaneously

```python
for chunk in graph.stream(
    inputs,
    stream_mode=["updates", "messages"],
    version="v2",
):
    if chunk["type"] == "updates":
        handle_state_update(chunk["data"])
    elif chunk["type"] == "messages":
        token, meta = chunk["data"]
        stream_to_user(token.content)
```

### Streaming from subgraphs

```python
for chunk in graph.stream(
    inputs,
    stream_mode="updates",
    subgraphs=True,   # include nested subgraph events
    version="v2",
):
    ns = chunk["ns"]   # () = root graph, ("node_name:<id>",) = subgraph
    print(f"Namespace: {ns}, Data: {chunk['data']}")
```

### Async streaming

```python
async def run():
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content="Hello")]},
        config={"configurable": {"thread_id": "1"}},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, _ = chunk["data"]
            if hasattr(token, "content") and token.content:
                print(token.content, end="", flush=True)
```

---

## 12. Key Imports Cheatsheet

```python
# ---- Core graph building ----
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState            # built-in chat state

# ---- State / reducers ----
from typing import Annotated
from typing_extensions import TypedDict
import operator                                       # operator.add for list append

# ---- LangGraph message reducer ----
from langgraph.graph.message import add_messages

# ---- Interrupts & commands ----
from langgraph.types import interrupt, Command

# ---- Custom streaming ----
from langgraph.config import get_stream_writer

# ---- Checkpointers ----
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver   # pip install langgraph-checkpoint-postgres
from langgraph.checkpoint.postgres import PostgresSaver            # sync variant

# ---- Pre-built agents ----
from langgraph.prebuilt import create_react_agent

# ---- Multi-agent libraries ----
from langgraph_supervisor import create_supervisor   # pip install langgraph-supervisor
from langgraph_swarm import create_swarm, create_handoff_tool  # pip install langgraph-swarm

# ---- LangChain core ----
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# ---- LLM providers ----
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
```

---

## Quick Patterns Reference

### Minimal ReAct agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, tools=[search])
result = agent.invoke({"messages": [{"role": "user", "content": "Who is Ada Lovelace?"}]})
print(result["messages"][-1].content)
```

### Graph with memory across turns

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-42"}}

# Turn 1
graph.invoke({"messages": [HumanMessage("Hi, I'm Alice")]}, config)
# Turn 2 — graph remembers previous messages
graph.invoke({"messages": [HumanMessage("What's my name?")]}, config)
```

### Parallel node execution (fan-out → fan-in)

```python
# Fan-out: supervisor dispatches to a and b simultaneously
builder.add_conditional_edges("supervisor", lambda s: ["node_a", "node_b"])
# Fan-in: merge waits for both
builder.add_edge(["node_a", "node_b"], "merge")
```

### invoke() with config

```python
result = graph.invoke(
    {"messages": [HumanMessage("Hello")]},
    config={
        "configurable": {"thread_id": "t1"},
        "callbacks": [langfuse_handler],
        "run_name": "my-run",
        "metadata": {"user_id": "u-123"},
    },
)
```

---

*Sources: [LangGraph GitHub](https://github.com/langchain-ai/langgraph) · [LangChain Reference](https://reference.langchain.com/python/langgraph) · [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph) · [PyPI langgraph 1.2.0](https://pypi.org/project/langgraph/) · [langgraph-checkpoint-postgres](https://pypi.org/project/langgraph-checkpoint-postgres/)*
