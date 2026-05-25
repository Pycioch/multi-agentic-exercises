# tau2-bench Comprehensive Documentation

> **Source**: [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)  
> **Also written as**: τ²-bench, tau2bench, τ-bench  
> **License**: MIT | **Python**: ≥3.12, <3.14  
> **Package manager**: `uv`

---

## 1. What is tau2-bench?

tau2-bench is a **simulation framework for evaluating conversational LLM agents in realistic, dual-control customer service environments**. Both the agent under test and the user simulator are AI-powered and can independently execute actions through domain-specific tool APIs.

Key innovations:
- **Dual-control architecture**: both agent and simulated user have tool access and act independently
- **Two communication modes**: text half-duplex (turn-based) and voice full-duplex (simultaneous streaming)
- **Multi-domain evaluation**: airline, retail, telecom, banking knowledge, mock
- **Pass^k metric**: statistically robust success rate across multiple trials
- **LiteLLM backend**: any supported LLM provider works

The framework is backed by a 40+ page whitepaper ([arXiv:2506.07982](https://arxiv.org/pdf/2506.07982)).

---

## 2. Installation

### Core (text domains: airline, retail, telecom, mock)

```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
uv sync
```

### With Extras

```bash
uv sync --extra voice        # Audio/voice full-duplex features (requires portaudio, ffmpeg)
uv sync --extra knowledge    # Banking knowledge domain with RAG pipelines
uv sync --extra gym          # Gymnasium RL interface for training
uv sync --extra dev          # pytest, ruff, pre-commit
uv sync --all-extras         # Everything
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

`.env.example` defines the following variables:

```env
# LLM providers
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=

# Voice (ElevenLabs TTS + Deepgram ASR)
ELEVENLABS_API_KEY=
DEEPGRAM_API_KEY=

# Voice persona IDs (ElevenLabs voice IDs)
TAU2_VOICE_ID_MATT_DELANEY=
TAU2_VOICE_ID_LISA_BRENNER=
TAU2_VOICE_ID_MILDRED_KAPLAN=
TAU2_VOICE_ID_ARJUN_ROY=
TAU2_VOICE_ID_WEI_LIN=
TAU2_VOICE_ID_MAMADOU_DIALLO=
TAU2_VOICE_ID_PRIYA_PATIL=
```

To use OpenRouter as the LLM backend:
```env
OPENAI_API_KEY=$YOUR_OPENROUTER_KEY
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

---

## 3. CLI Commands

### `tau2 intro`
Show an overview of tau2-bench, available domains, commands, and examples.

```bash
tau2 intro
```

### `tau2 run` — Main Evaluation Command

```bash
tau2 run --domain DOMAIN --agent-llm MODEL --user-llm MODEL \
         --num-trials N --num-tasks N
```

**Core arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--domain` | str | required | Domain: `airline`, `retail`, `telecom`, `banking_knowledge`, `mock` |
| `--agent` | str | `gpt-4.1` | Agent implementation name (registered in registry) |
| `--agent-llm` | str | `gpt-4.1` | LLM model for agent |
| `--agent-llm-args` | JSON | `{"temperature": 0.7}` | Extra LLM kwargs for agent |
| `--user` | str | `gpt-4.1` | User simulator implementation name |
| `--user-llm` | str | `gpt-4.1` | LLM model for user simulator |
| `--user-llm-args` | JSON | `{"temperature": 0.5}` | Extra LLM kwargs for user |
| `--num-trials` | int | 1 | Number of repetitions per task (for Pass^k) |
| `--num-tasks` | int | None | Limit number of tasks to run |
| `--max-steps` | int | 50 | Max conversation turns per simulation |
| `--max-errors` | int | 3 | Tool error tolerance before abort |
| `--timeout` | float | None | Wallclock time limit per simulation (seconds) |
| `--seed` | int | 42 | Random seed |
| `--max-concurrency` | int | 10 | Parallel simulations |
| `--save-to` | str | auto | Results save path |
| `--log-level` | str | `INFO` | Logging level |
| `--verbose-logs` | flag | False | Detailed logging (LLM calls, ticks) |
| `--max-retries` | int | 3 | Retry attempts on failure |
| `--retry-delay` | float | 1.0 | Seconds between retries |
| `--auto-resume` | flag | False | Resume checkpoint without prompting |
| `--auto-review` | flag | False | Run LLM review post-simulation |
| `--review-mode` | str | `full` | Review scope: `full` or `user-only` |
| `--hallucination-retries` | int | 3 | Retries when hallucination detected |

**Voice / full-duplex arguments** (requires `--extra voice`):

| Argument | Default | Description |
|---|---|---|
| `--audio-native` | False | Enable full-duplex voice mode |
| `--audio-native-provider` | `openai` | Provider: `openai`, `gemini`, `xai`, `livekit` |
| `--audio-native-model` | auto | Override model for provider |
| `--tick-duration` | 0.05 | Tick interval in seconds |
| `--max-steps-seconds` | 120 | Max conversation duration (seconds) |
| `--speech-complexity` | `regular` | `regular` or `complex` |
| `--audio-debug` | False | Enable audio timing analysis |
| `--audio-taps` | False | Save WAV files at pipeline stages |

**Example runs:**

```bash
# Basic text evaluation
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 \
         --num-trials 3 --num-tasks 5

# Custom agent
tau2 run --domain retail --agent my_agent --agent-llm claude-3-5-sonnet-20241022 \
         --user-llm gpt-4.1 --num-trials 5

# Banking knowledge with RAG
tau2 run --domain banking_knowledge --agent-llm gpt-4.1 --user-llm gpt-4.1 \
         --retrieval-config openai_embeddings --num-trials 3

# Voice evaluation
tau2 run --domain airline --audio-native --audio-native-provider openai \
         --audio-native-model gpt-4o-realtime-preview --num-tasks 10
```

### `tau2 make test`

Run the test suite (uses `pytest` under the hood, requires `--extra dev`):

```bash
make test
# or
uv run pytest tests/
```

### Other CLI Commands

| Command | Description |
|---|---|
| `tau2 view` | Browse and inspect saved simulation results |
| `tau2 play` | Interactive manual mode — play the agent yourself |
| `tau2 domain <domain>` | Show ReDoc API documentation for a domain |
| `tau2 review` | Run LLM-based review on existing results |
| `tau2 evaluate-trajs` | Re-evaluate trajectories and recompute rewards |
| `tau2 leaderboard` | Display current leaderboard |
| `tau2 submit prepare` | Prepare a leaderboard submission |
| `tau2 submit validate` | Validate a submission directory |
| `tau2 check-data` | Verify data directory setup |
| `tau2 start` | Start background servers |

### RunExperiments Mode

There is no separate "RunExperiments" CLI mode. Experiment-level configuration is done programmatically via `TextRunConfig` or `VoiceRunConfig` objects passed to `run_domain()`. See Section 8 (Runner) for details.

---

## 4. HalfDuplexAgent — Base Class Interface

The `HalfDuplexAgent` is the base class for all text-based (turn-by-turn) agents.

### Class Hierarchy

```
HalfDuplexParticipant[ValidAgentInputMessage, AssistantMessage, AgentState]
  └── HalfDuplexAgent[AgentState]          ← your base class
        └── LLMAgent                       ← built-in reference implementation
```

### Constructor

```python
def __init__(self, tools: list[Tool], domain_policy: str):
    super().__init__()
    self.tools = tools
    self.domain_policy = domain_policy
```

Both `tools` (environment API endpoints) and `domain_policy` (policy text string) are required.

### Abstract Methods to Implement

```python
# Called once at the start of a conversation with optional history
def get_init_state(self, message_history: list) -> AgentState:
    ...

# Called each turn to produce the agent's response
def generate_next_message(
    self,
    message: UserMessage | ToolMessage | MultiToolMessage,
    state: AgentState,
) -> tuple[AssistantMessage, AgentState]:
    ...
```

### Optional Method

```python
def stop(
    self,
    message: Optional[ValidAgentInputMessage] = None,
    state: Optional[AgentState] = None,
) -> None:
    """Called when the simulation ends. Override for cleanup."""
    pass
```

### Minimal Custom Agent Example

```python
from tau2 import HalfDuplexAgent, AssistantMessage
from tau2.data_model.message import UserMessage, ToolMessage, MultiToolMessage
from tau2.environment.tool import Tool
from pydantic import BaseModel

class MyAgentState(BaseModel):
    system_messages: list
    messages: list
    # add custom fields here

class MyAgent(HalfDuplexAgent[MyAgentState]):
    def __init__(self, tools: list[Tool], domain_policy: str):
        super().__init__(tools=tools, domain_policy=domain_policy)

    def get_init_state(self, message_history: list) -> MyAgentState:
        return MyAgentState(
            system_messages=[{"role": "system", "content": self.domain_policy}],
            messages=list(message_history),
        )

    def generate_next_message(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: MyAgentState,
    ) -> tuple[AssistantMessage, MyAgentState]:
        # append incoming message to history
        state.messages.append(message)
        # ... call LLM, handle tool calls ...
        response = AssistantMessage.text(content="Hello, how can I help?")
        return response, state
```

---

## 5. LLMConfigMixin

`LLMConfigMixin` is a mixin that adds LLM model configuration to any participant class (agent or user simulator).

### Location

```python
from tau2.agent.base.llm_config import LLMConfigMixin
# or via top-level:
from tau2 import LLMConfigMixin
```

### What It Provides

```python
class LLMConfigMixin:
    llm: str          # LLM model name (e.g., "gpt-4.1", "claude-3-5-sonnet-20241022")
    llm_args: dict    # Additional kwargs passed to the LLM (e.g., temperature, max_tokens)

    def __init__(self, *args, llm: str, llm_args: Optional[dict] = None, **kwargs):
        # sets self.llm and deep-copies llm_args (defaults to {})
        ...

    def set_seed(self, seed: int) -> None:
        """Set/override the random seed in llm_args."""
        ...
```

### Usage Pattern

Mix it in before your base class (standard Python MRO):

```python
class MyLLMAgent(LLMConfigMixin, HalfDuplexAgent[MyState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
```

The built-in `LLMAgent` already uses this pattern:
```python
class LLMAgent(LLMConfigMixin, HalfDuplexAgent[LLMAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        ...
```

---

## 6. Tool Interface

### Location

```python
from tau2.environment.tool import Tool, BaseTool, as_tool
```

### `BaseTool` (Abstract)

```python
class BaseTool(BaseModel, ABC):
    name: str  # identifier for the tool

    @property
    @abstractmethod
    def openai_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def _call(self, *args, **kwargs) -> Any: ...

    def __call__(self, *args, **kwargs) -> Any:
        return self._call(*args, **kwargs)
```

### `Tool` (Concrete — wraps a Python function)

```python
class Tool(BaseTool):
    short_desc: str        # brief description shown to the LLM
    long_desc: str         # full documentation
    params: type[BaseModel]   # Pydantic model for input parameters (auto-generated)
    returns: type[BaseModel]  # Pydantic model for output (auto-generated)
    raises: list[dict]     # exceptions: [{"type": "...", "description": "..."}]
    examples: list[str]    # usage examples
    info: dict             # additional metadata

    def __init__(self, func: Callable, use_short_desc: bool = False, **predefined: Any):
        """Create a Tool from a Python function.
        predefined: keyword args to pre-bind to the function.
        """
```

### `as_tool` Factory Function

```python
def as_tool(func: Callable, **predefined_kwargs) -> Tool:
    """Wrap a function with pre-bound arguments into a Tool."""
    return Tool(func=func, **predefined_kwargs)
```

### Creating a Tool from a Python Function

```python
from tau2.environment.tool import as_tool

def get_reservation(reservation_id: str, db) -> dict:
    """Retrieve reservation details.
    
    Args:
        reservation_id: The reservation ID to look up.
        db: The database instance.
    
    Returns:
        dict: Reservation details including flights and passengers.
    """
    return db.find(reservation_id)

# Pre-bind the db argument; the LLM only sees reservation_id
tool = as_tool(get_reservation, db=my_database_instance)
```

### LangChain Tool Conversion

tau2-bench does **not** provide a built-in `langchain_to_tool` utility. To use LangChain tools, wrap them manually:

```python
from tau2.environment.tool import Tool
from langchain.tools import BaseTool as LCTool

def lc_to_tau2(lc_tool: LCTool) -> Tool:
    def wrapped(**kwargs):
        return lc_tool.run(kwargs)
    wrapped.__name__ = lc_tool.name
    wrapped.__doc__ = lc_tool.description
    return as_tool(wrapped)
```

### How Agents Receive Tools

```python
# Inside the environment (e.g., airline domain):
env = get_environment()
tools: list[Tool] = env.get_tools()    # list of Tool objects
policy: str = env.get_policy()         # policy text string

agent = MyAgent(tools=tools, domain_policy=policy)
```

---

## 7. Data Model — Message Types

### Location

```python
from tau2.data_model.message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolMessage,
    MultiToolMessage,
    EnvironmentMessage,  # type alias: ToolMessage | MultiToolMessage
)
```

### `SystemMessage`

```python
# System prompt / instructions
msg = SystemMessage(content="You are a helpful airline agent.")
# Fields: role="system", content: str, turn_index: int, timestamp
```

### `UserMessage`

```python
# Text message from the user
msg = UserMessage.text(content="I want to cancel my reservation.")

# Voice message (base64-encoded audio)
msg = UserMessage.voice(audio_content=b64_audio_bytes, audio_format="pcm")

# Fields: role, content, is_audio, tool_calls, timestamps,
#         audio_format, audio_content, audio_path, audio effects,
#         streaming metadata (chunk_id, utterance_ids)
```

### `AssistantMessage`

```python
# Text response from the agent
msg = AssistantMessage.text(content="I can help you cancel your reservation.")

# With tool calls
msg = AssistantMessage.text(
    content="Let me look that up.",
    tool_calls=[ToolCall(id="tc_1", name="get_reservation_details",
                         arguments={"reservation_id": "ABC123"}, requestor="assistant")]
)

# Voice response
msg = AssistantMessage.voice(audio_content=b64_bytes, audio_format="pcm")
```

### `ToolCall`

```python
# Represents a function call issued by agent or user
tc = ToolCall(
    id="call_abc123",
    name="cancel_reservation",
    arguments={"reservation_id": "XYZ789"},
    requestor="assistant",  # "assistant" or "user"
)

# Parse from string
tc = ToolCall.from_string(json_string)
```

### `ToolMessage`

```python
# Response from tool execution
tm = ToolMessage(
    id="call_abc123",
    role="tool",
    content='{"status": "cancelled", "refund": 250.00}',
    requestor="assistant",
    error=False,   # True if the tool raised an exception
    turn_index=3,
)
```

### `MultiToolMessage`

```python
# Container for multiple tool responses (when agent calls multiple tools at once)
mtm = MultiToolMessage(
    role="tool",
    messages=[tool_message_1, tool_message_2],
)
```

### `EnvironmentMessage` (Type Alias)

```python
EnvironmentMessage = ToolMessage | MultiToolMessage
```

### Valid Agent Input

The agent's `generate_next_message` accepts:
```python
ValidAgentInputMessage = UserMessage | ToolMessage | MultiToolMessage
```

---

## 8. Runner — TextRunConfig, build_orchestrator, get_tasks, run_simulation

### Module Structure

```
tau2.runner/
├── simulation.py    # Layer 1: run_simulation()
├── build.py         # Layer 2: build_* component constructors
├── batch.py         # Layer 3: run_domain(), run_tasks(), run_single_task()
└── helpers.py       # get_tasks(), load_tasks(), make_run_name(), get_info()
```

### Imports

```python
from tau2.runner import (
    # Layer 1
    run_simulation,
    # Layer 2
    build_environment, build_agent, build_user, build_orchestrator,
    build_text_orchestrator, build_voice_orchestrator,
    # Layer 3
    run_domain, run_tasks, run_single_task,
    # Helpers
    get_tasks, load_tasks, load_task_splits, get_options,
    get_environment_info, make_run_name, get_info,
)
```

---

### `TextRunConfig`

Configuration object for half-duplex (text/turn-based) simulations.

```python
from tau2 import TextRunConfig

config = TextRunConfig(
    # === Required ===
    domain="airline",           # domain name
    
    # === Agent ===
    agent="llm_agent",          # registered agent name (default: "llm_agent")
    llm_agent="gpt-4.1",        # LLM model for agent
    llm_args_agent={"temperature": 0.7},
    
    # === User Simulator ===
    user="user_simulator",      # registered user name (default: "user_simulator")
    llm_user="gpt-4.1",
    llm_args_user={"temperature": 0.5},
    
    # === Task Selection ===
    task_set_name="airline",    # which task set to load (usually = domain)
    task_split_name="base",     # split variant (None = all tasks)
    task_ids=None,              # specific task IDs to run, or None for all
    num_tasks=None,             # limit number of tasks
    
    # === Execution ===
    num_trials=3,               # repetitions per task (for Pass^k)
    max_steps=50,               # max turns per conversation
    max_errors=3,               # tool errors before aborting
    timeout=None,               # seconds per simulation
    seed=42,
    max_concurrency=10,
    log_level="INFO",
    verbose_logs=False,
    
    # === Output ===
    save_to=None,               # auto-generated path if None
    
    # === Retry/Resume ===
    max_retries=3,
    retry_delay=1.0,
    auto_resume=False,
    auto_review=False,
    review_mode="full",
    hallucination_retries=3,
    
    # === Protocol ===
    enforce_communication_protocol=False,
    text_streaming_config=None,
    
    # === Knowledge (banking_knowledge domain) ===
    retrieval_config=None,       # e.g., "openai_embeddings", "bm25", "grep_only"
    retrieval_config_kwargs={},
)
```

**Properties available on the config:**
- `config.effective_agent` — resolved agent name
- `config.effective_user` — resolved user name
- `config.effective_max_steps` — resolved max steps
- `config.effective_agent_model` — LLM model string
- `config.effective_agent_provider` — returns `None` for text mode

---

### `get_tasks()`

```python
from tau2.runner import get_tasks

tasks = get_tasks(
    task_set_name="airline",        # required: which task set to load
    task_split_name="base",         # optional: split name (e.g., "base")
    task_ids=["task_001", "task_002"],  # optional: filter to specific IDs
    num_tasks=10,                   # optional: limit count
)
# Returns: list[Task]
```

---

### `build_orchestrator()`

Constructs a fully configured orchestrator from a config and a task. Dispatches to `build_text_orchestrator` or `build_voice_orchestrator` based on config type.

```python
from tau2.runner import build_orchestrator

orchestrator = build_orchestrator(
    config=config,                           # TextRunConfig or VoiceRunConfig
    task=task,                               # Task object
    seed=42,                                 # optional
    simulation_id="sim_001",                 # optional
    user_voice_settings=None,                # optional, VoiceSettings
    user_persona_config=None,                # optional, PersonaConfig
    hallucination_feedback=None,             # optional str feedback for retry
    audio_taps_dir=None,                     # optional Path for WAV taps
)
```

**Text-specific builder:**

```python
from tau2.runner import build_text_orchestrator

orchestrator = build_text_orchestrator(
    config=config,            # TextRunConfig
    task=task,
    seed=None,
    simulation_id=None,
    user_persona_config=None,
)
# Returns: Orchestrator (half-duplex)
```

**Low-level builders:**

```python
from tau2.runner import build_environment, build_agent, build_user

env = build_environment(domain="airline")
agent = build_agent(agent_name="my_agent", environment=env)
user = build_user(user_name="user_simulator", environment=env, task=task)
```

---

### `run_simulation()`

Layer 1 — executes a pre-built orchestrator. No registry, no config parsing, no side effects.

```python
from tau2.runner import run_simulation
from tau2.evaluator.evaluator import EvaluationType

simulation_run = run_simulation(
    orchestrator=orchestrator,
    evaluation_type=EvaluationType.ALL,   # default
    env_kwargs=None,                      # e.g., {"retrieval_variant": "bm25"}
)
# Returns: SimulationRun (with .reward_info attached)
print(simulation_run.reward_info.reward)  # 0.0 or 1.0
```

---

### Layer 3 — Batch Operations

#### `run_domain()` — Full pipeline

```python
from tau2.runner import run_domain

results = run_domain(config=config)
# Loads tasks, filters, runs batch, displays metrics
# Returns: Results
```

#### `run_tasks()` — Run a task list

```python
from tau2.runner import run_tasks
from tau2.evaluator.evaluator import EvaluationType

results = run_tasks(
    config=config,
    tasks=tasks,
    save_path=None,                        # optional Path
    save_dir=None,                         # optional Path
    evaluation_type=EvaluationType.ALL,
    console_display=True,
    results_format="json",
)
```

#### `run_single_task()` — Single task

```python
from tau2.runner import run_single_task

sim_run = run_single_task(
    config=config,
    task=task,
    seed=42,
    evaluation_type=EvaluationType.ALL,
    save_dir=None,
    user_voice_settings=None,
    user_persona_config=None,
    verbose_logs=False,
    audio_debug=False,
    audio_taps=False,
    auto_review=False,
    review_mode="full",
    hallucination_feedback=None,
)
```

---

### Complete Programmatic Example

```python
from tau2 import TextRunConfig
from tau2.runner import get_tasks, build_orchestrator, run_simulation

# 1. Configure
config = TextRunConfig(
    domain="airline",
    agent="my_agent",
    llm_agent="gpt-4.1",
    llm_user="gpt-4.1",
    num_trials=3,
)

# 2. Load tasks
tasks = get_tasks(task_set_name="airline", task_split_name="base", num_tasks=5)

# 3. Run one task
task = tasks[0]
orchestrator = build_orchestrator(config=config, task=task, seed=42)
result = run_simulation(orchestrator)
print(f"Reward: {result.reward_info.reward}")  # 0.0 or 1.0

# 4. Or run all tasks at once
from tau2.runner import run_domain
results = run_domain(config)
print(results)
```

---

## 9. Registry — Registering an Agent Factory

All agents, domains, tasks, and users are registered in `src/tau2/registry.py` via a global `registry` singleton.

### Import

```python
from tau2 import registry           # global Registry instance
from tau2.registry import Registry  # the class itself
```

### Register an Agent Factory

```python
def create_my_agent(tools, domain_policy, **kwargs):
    """
    Factory function called by the eval framework.
    kwargs contains CLI/config args: llm, llm_args, seed, etc.
    """
    llm = kwargs.get("llm", "gpt-4.1")
    llm_args = kwargs.get("llm_args", {})
    return MyAgent(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)

registry.register_agent_factory(
    factory=create_my_agent,
    name="my_agent",                        # name used in --agent CLI flag
    task_filter=None,                       # optional: Callable[[Task], bool]
    metadata=None,                          # optional: {"solo_mode": True} etc.
)
```

### Registry API

```python
# Registration
registry.register_agent_factory(factory, name, task_filter=None, metadata=None)
registry.register_domain(get_environment, name)
registry.register_tasks(get_tasks, name, get_task_splits=None)
registry.register_user(user_constructor, name=None)

# Retrieval
registry.get_agent_factory(name)              # -> Optional[Callable]
registry.get_agent_task_filter(name)          # -> Optional[Callable[[Task], bool]]
registry.get_agent_metadata(name, key, default=None)
registry.get_user_constructor(name)           # -> type
registry.get_env_constructor(name)            # -> Callable[[], Environment]
registry.get_tasks_loader(name)               # -> Callable[[Optional[str]], list[Task]]
```

### Built-in Registered Agents

```python
# Default text agent (uses ground-truth user simulator)
"llm_agent"       # LLMGTAgent — standard evaluation agent

# Solo mode (no user simulator, agent acts alone)
"llm_agent_solo"  # LLMSoloAgent — metadata={"solo_mode": True}
```

---

## 10. Domains

Results are saved to `data/simulations/`. Domain data lives in `data/tau2/domains/<domain>/`:
- `tasks.json` — task definitions
- `db.json` — initial database state
- `policy.md` — agent policy text

### `mock`
Lightweight test domain for development and debugging. No real tools.

### `airline`
**Scenario**: Customer service agent for a fictional airline.  
**Tasks**: ~50 tasks covering booking, cancellation, modification, baggage.  
**Default split**: `"base"`  
**Solo mode**: Not supported.

**Agent tools** (`AirlineTools`):
| Tool | Type | Description |
|---|---|---|
| `list_all_airports` | read | Return all airport IATA codes |
| `search_direct_flight` | read | Find nonstop flights |
| `search_onestop_flight` | read | Find connecting flights |
| `get_reservation_details` | read | Full reservation info |
| `get_user_details` | read | User profile + reservations |
| `get_flight_status` | read | Flight availability |
| `book_reservation` | write | Create new booking |
| `cancel_reservation` | write | Cancel + refund |
| `update_reservation_flights` | write | Change flights |
| `update_reservation_passengers` | write | Change passenger info |
| `update_reservation_baggages` | write | Adjust baggage |
| `send_certificate` | write | Issue travel certificate |
| `calculate` | utility | Math expressions |
| `transfer_to_human_agents` | utility | Escalate to human |

### `retail`
**Scenario**: E-commerce customer service (order management, returns).  
**Includes 50 product types.**

**Agent tools** (`RetailTools`):
| Tool | Type | Description |
|---|---|---|
| `find_user_id_by_name_zip` | read | Look up user by name + zip |
| `find_user_id_by_email` | read | Look up user by email |
| `get_order_details` | read | Order status and items |
| `get_product_details` | read | Product info |
| `get_item_details` | read | Item variant info |
| `get_user_details` | read | User profile + orders |
| `list_all_product_types` | read | All 50 product types |
| `cancel_pending_order` | write | Cancel + refund |
| `modify_pending_order_address` | write | Update shipping address |
| `modify_pending_order_items` | write | Exchange items |
| `modify_pending_order_payment` | write | Change payment |
| `modify_user_address` | write | Update default address |
| `exchange_delivered_order_items` | write | Exchange delivered items |
| `return_delivered_order_items` | write | Initiate return |
| `calculate` | utility | Math expressions |
| `transfer_to_human_agents` | utility | Escalate |

### `telecom`
**Scenario**: Telecommunications troubleshooting — the only domain with **dual-control** (both agent and user have tools).  
**User tools simulate a real phone** (the simulated user runs diagnostic commands on their device).

**Agent tools** (`TelecomTools`):
| Tool | Description |
|---|---|
| `get_customer_by_phone` | Find customer by phone number |
| `get_customer_by_id` | Find customer by ID |
| `get_customer_by_name` | Find by name + DOB |
| `get_details_by_id` | Get Customer/Line/Device/Bill/Plan |
| `get_bills_for_customer` | List bills (most recent first) |
| `get_data_usage` | Current billing cycle data usage |
| `suspend_line` | Suspend a line (max 6 months) |
| `resume_line` | Reactivate a line |
| `send_payment_request` | Send payment request for a bill |
| `enable_roaming` | Activate international roaming |
| `disable_roaming` | Deactivate roaming |
| `refuel_data` | Add data to a line |
| `transfer_to_human_agents` | Escalate |

**User tools** (phone simulator — `user_tools.py`): `check_status_bar`, `check_network_status`, `check_network_mode_preference`, `run_speed_test`, `set_network_mode_preference`, `toggle_airplane_mode`, `toggle_data`, `toggle_roaming`, `toggle_data_saver_mode`, `check_data_restriction_status`, `check_sim_status`, `reseat_sim_card`, `unseat_sim_card`, `check_apn_settings`, `set_apn_settings`, `reset_apn_settings`, `check_wifi_status`, `toggle_wifi`, `check_wifi_calling_status`, `toggle_wifi_calling`, `check_vpn_status`, `connect_vpn`, `disconnect_vpn`, `check_installed_apps`, `check_app_status`, `check_app_permissions`, `grant_app_permission`, `can_send_mms`, `reboot_device`.

### `banking_knowledge`
**Scenario**: Knowledge-retrieval customer service for a bank. Agent must answer questions using policy documents + database.  
**Requires**: `--extra knowledge`  
**Scale**: 97 tasks, 698 policy documents.

**Retrieval config** (`--retrieval-config`):

| Config name | Toolkit | Description |
|---|---|---|
| `no_knowledge` | `KnowledgeToolsPlain` | No retrieval |
| `full_kb` | `KnowledgeToolsPlain` | Full KB access (no search) |
| `golden_retrieval` | `KnowledgeToolsPlain` | Oracle retrieval |
| `bm25` | `KnowledgeToolsWithKBSearch` | BM25 search |
| `qwen_embeddings` | `KnowledgeToolsWithKBSearch` | Dense search (Qwen) |
| `openai_embeddings` | `KnowledgeToolsWithKBSearch` | Dense search (OpenAI) |
| `grep_only` | `KnowledgeToolsWithGrep` | Shell grep only |
| `bm25_grep` | `KnowledgeToolsWithKBSearchAndGrep` | BM25 + grep |
| `qwen_embeddings_grep` | `KnowledgeToolsWithKBSearchAndGrep` | Dense + grep |
| `openai_embeddings_grep` | `KnowledgeToolsWithKBSearchAndGrep` | Dense + grep |
| `terminal_use` | `KnowledgeToolsWithShell` | Shell (read-only) |
| `terminal_use_write` | `KnowledgeToolsWithShell` | Shell (read-write) |
| `AllTools` | `KnowledgeToolsAllTools` | BM25 + dense + shell |

```bash
tau2 run --domain banking_knowledge --retrieval-config openai_embeddings \
         --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 3
```

---

## 11. Reward / Scoring

### Reward Components

The final reward is **binary: 0.0 (fail) or 1.0 (pass)** per simulation.

The `reward_basis` field on each `Task` controls which components gate the reward. The final reward is the **product** of all components listed in `reward_basis`. Any component returning 0.0 causes the whole task to fail.

| Component | Enum | What it checks |
|---|---|---|
| `DB` | `reward_basis.DB` | Predicted DB state matches target (from replaying reference actions) |
| `ENV_ASSERTION` | `reward_basis.ENV_ASSERTION` | All environment assertions pass |
| `COMMUNICATE` | `reward_basis.COMMUNICATE` | Agent messages contain required substrings |
| `NL_ASSERTION` | `reward_basis.NL_ASSERTION` | LLM judges natural language assertions as true |
| `ACTION` | `reward_basis.ACTION` | Agent tool calls match reference trajectory exactly |

Default `reward_basis`: `[DB, COMMUNICATE]`

Components NOT in `reward_basis` still run diagnostically but don't affect the score.

### `EvaluationType` Enum

```python
from tau2.evaluator.evaluator import EvaluationType

EvaluationType.ENV                          # DB + ENV_ASSERTION only
EvaluationType.COMMUNICATE                  # COMMUNICATE only
EvaluationType.ACTION                       # ACTION only
EvaluationType.ALL                          # respects task's reward_basis (default)
EvaluationType.NL_ASSERTIONS                # NL_ASSERTION only (WIP)
EvaluationType.ALL_WITH_NL_ASSERTIONS       # ALL + force NL_ASSERTION
EvaluationType.ALL_IGNORE_BASIS             # all four components multiplied, ignores reward_basis
EvaluationType.ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS  # all four, ignores reward_basis
```

### Pass^k Metric

Pass^k is the **primary aggregate metric**. It estimates the probability that at least k out of num_trials simulation runs succeed for a given task, then averages across all tasks.

**Formula:**

```
pass^k = C(success_count, k) / C(num_trials, k)
```

where `C(n, k)` is the binomial coefficient ("n choose k").

**Implementation:**

```python
from tau2.metrics.agent_metrics import pass_hat_k

score = pass_hat_k(
    num_trials=5,        # total runs per task
    success_count=3,     # successful runs
    k=1,                 # typically k=1 for standard evaluation
)
# → 0.6

# A trial is successful when:
# (1 - 1e-6) <= reward <= (1 + 1e-6)  (i.e., reward ≈ 1.0)
```

**Overall Pass^k** is the **mean** of per-task Pass^k values across all tasks.

**Recommended usage**: run with `--num-trials 5` or more to get meaningful Pass^k statistics.

---

## 12. Environment Variables

### Required

| Variable | When needed |
|---|---|
| `OPENAI_API_KEY` | Any OpenAI model (or OpenRouter proxy) |
| `ANTHROPIC_API_KEY` | Claude models via Anthropic |
| `OPENROUTER_API_KEY` | OpenRouter multi-provider access |

### Voice / TTS / ASR

| Variable | When needed |
|---|---|
| `ELEVENLABS_API_KEY` | Voice TTS synthesis |
| `DEEPGRAM_API_KEY` | Voice ASR transcription |

### Voice Persona IDs (ElevenLabs)

| Variable | Persona |
|---|---|
| `TAU2_VOICE_ID_MATT_DELANEY` | Voice persona |
| `TAU2_VOICE_ID_LISA_BRENNER` | Voice persona |
| `TAU2_VOICE_ID_MILDRED_KAPLAN` | Voice persona |
| `TAU2_VOICE_ID_ARJUN_ROY` | Voice persona |
| `TAU2_VOICE_ID_WEI_LIN` | Voice persona |
| `TAU2_VOICE_ID_MAMADOU_DIALLO` | Voice persona |
| `TAU2_VOICE_ID_PRIYA_PATIL` | Voice persona |

### OpenRouter Proxy Pattern

```env
OPENAI_API_KEY=your_openrouter_key_here
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

---

## 13. Key Imports Cheatsheet

```python
# === Top-level convenience imports ===
from tau2 import (
    # Agents
    HalfDuplexAgent,
    FullDuplexAgent,
    LLMAgent,
    LLMSoloAgent,
    LLMConfigMixin,
    
    # Configuration
    TextRunConfig,
    VoiceRunConfig,
    RunConfig,          # Union[TextRunConfig, VoiceRunConfig]
    BaseRunConfig,
    
    # Data models
    SimulationRun,
    Task,
    
    # Infrastructure
    Environment,
    Orchestrator,
    CommunicationMode,
    
    # Evaluation
    EvaluationType,
    evaluate_simulation,
    
    # Registry
    registry,
    Registry,
    
    # Runner (high-level)
    run_domain,
    
    # Users
    UserSimulator,
    HalfDuplexUser,
    FullDuplexUser,
    
    # Display
    ConsoleDisplay,
    MarkdownDisplay,
)

# === Runner (all layers) ===
from tau2.runner import (
    run_simulation,           # Layer 1
    build_orchestrator,       # Layer 2
    build_text_orchestrator,
    build_voice_orchestrator,
    build_environment,
    build_agent,
    build_user,
    run_domain,               # Layer 3
    run_tasks,
    run_single_task,
    get_tasks,                # Helpers
    load_tasks,
    load_task_splits,
    get_options,
    get_environment_info,
    make_run_name,
    get_info,
)

# === Data model ===
from tau2.data_model.message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolMessage,
    MultiToolMessage,
    EnvironmentMessage,   # = ToolMessage | MultiToolMessage
)
from tau2.data_model.simulation import (
    TextRunConfig,
    VoiceRunConfig,
    RunConfig,
    BaseRunConfig,
    SimulationRun,
)
from tau2.data_model.tasks import Task

# === Tools ===
from tau2.environment.tool import Tool, BaseTool, as_tool

# === Agent bases ===
from tau2.agent.base_agent import HalfDuplexAgent, FullDuplexAgent
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent

# === Registry ===
from tau2.registry import registry, Registry

# === Evaluator ===
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

# === Metrics ===
from tau2.metrics.agent_metrics import pass_hat_k
```

---

## 14. Complete Custom Agent Example

```python
# my_agent.py — register a custom agent in the tau2 framework

from typing import Optional
from pydantic import BaseModel
from tau2 import HalfDuplexAgent, LLMConfigMixin, registry
from tau2.data_model.message import (
    SystemMessage, UserMessage, ToolMessage, MultiToolMessage, AssistantMessage,
)
from tau2.environment.tool import Tool


# 1. Define agent state
class MyAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list

    
# 2. Implement the agent
class MyAgent(LLMConfigMixin, HalfDuplexAgent[MyAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "gpt-4.1",
        llm_args: Optional[dict] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args or {},
        )

    def get_init_state(self, message_history: list) -> MyAgentState:
        system_prompt = f"You are a helpful customer service agent.\n\n{self.domain_policy}"
        return MyAgentState(
            system_messages=[SystemMessage(content=system_prompt)],
            messages=list(message_history),
        )

    def generate_next_message(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: MyAgentState,
    ) -> tuple[AssistantMessage, MyAgentState]:
        # Append incoming message
        state.messages.append(message)
        
        # Build messages for LLM call
        all_messages = [*state.system_messages, *state.messages]
        
        # Call LLM (pseudocode — use your preferred client)
        tool_schemas = [t.openai_schema for t in self.tools]
        response = call_llm(
            model=self.llm,
            messages=all_messages,
            tools=tool_schemas,
            **self.llm_args,
        )
        
        assistant_msg = AssistantMessage.text(content=response.content)
        state.messages.append(assistant_msg)
        return assistant_msg, state


# 3. Factory function
def create_my_agent(tools, domain_policy, **kwargs):
    return MyAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "gpt-4.1"),
        llm_args=kwargs.get("llm_args", {}),
    )


# 4. Register (must run at import time — add to registry.py or call at module load)
registry.register_agent_factory(create_my_agent, "my_agent")
```

Then run:
```bash
tau2 run --domain airline --agent my_agent --agent-llm gpt-4.1 \
         --user-llm gpt-4.1 --num-trials 3 --num-tasks 10
```

---

## Sources

- [GitHub: sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- [DeepWiki: sierra-research/tau2-bench](https://deepwiki.com/sierra-research/tau2-bench)
- [Agent README](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/agent/README.md)
- [Runner README](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/runner/README.md)
- [Domains README](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/domains/README.md)
- [arXiv paper: τ²-Bench](https://arxiv.org/pdf/2506.07982)
- [Quesma blog post](https://quesma.com/blog/tau2-from-llm-benchmark-to-blueprint-for-testing-ai-agents/)
