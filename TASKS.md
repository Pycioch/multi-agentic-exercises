# Multi-Agent Exercises — Task Sheet

Each exercise has a starter file in its own folder.
The file contains the task description, required libraries, node structure, and mock data.
**Your job: build the graph from scratch using only the provided data.**

Each solution must trace every LLM call through Langfuse (`lf` handler is already set up).
Use `Command(goto=...)` for all routing decisions — not `add_conditional_edges`.

---

## 00 — Minimal LangChain Invocation

**File:** `00_simple_langchain_invocation/exercise.py`

Create the simplest possible LangChain program: initialize `ChatOpenAI`, send one prompt with `invoke()`, and print `response.content`.
Keep it to one call and one output line so participants can validate `.env` and model access before graph-based patterns.

---

## 01 — Sequential Pipeline

**File:** `01_sequential_pipeline/exercise.py`

Build a 4-node pipeline that processes a randomly selected research snippet.
Nodes run in a fixed order with no branching: Extractor → Writer → Fact-Checker → Publisher.
Each node reads only from the previous node's output in state.
Print each node's output as the pipeline runs.

---

## 02 — Tool Use / ReAct Agent

**File:** `02_tool_use_agent/exercise.py`

Build a `create_agent()` ReAct loop with 3 tools: `search_runbook`, `check_service_health`, `read_recent_logs`.
Run it against a randomly selected incident from `INCIDENTS`.
The agent must chain tool calls as needed and end with a concrete recommendation.
This is the only exercise where "agent" is the right word — `create_agent()` builds the full loop for you.

---

## 03 — Reflection Loop

**File:** `03_reflection_loop/exercise.py`

Build a 2-node loop: `generate` → `critique`.
`critique` scores the draft 1–10 and returns `Command(goto="generate")` or `Command(goto=END)`.
Stop when score ≥ 8 or after 4 iterations — whichever comes first.
Parse the score with regex. Pick a task at random from `CODING_TASKS`.

---

## 04 — Evaluator-Optimizer

**File:** `04_evaluator_optimizer/exercise.py`

Build a 2-node loop: `generate` → `evaluate`.
`evaluate` is an **independent judge** — it scores on 4 rubric dimensions (Clarity, Specificity, SEO, CTA), 25 pts each.
Stop when total ≥ 82/100 or after 3 iterations.
`generate` must receive the evaluator's structured feedback on each rerun.
Pick a product brief at random from `PRODUCT_BRIEFS`.

---

## 05 — Router / Dispatcher

**File:** `05_router_dispatcher/exercise.py`

Build a `router` node that classifies an incoming message into one of 4 departments
and returns `Command(goto=department)`.
Each department (`billing`, `technical`, `account`, `sales`) is a separate node
with its own knowledge base from `SPECIALIST_KB`.
After the specialist responds the graph ends — no aggregation.
Pick a message at random from `SUPPORT_MESSAGES`.

---

## 06 — Plan-and-Execute with Replanning

**File:** `06_planning_execute/exercise.py`

Build 4 nodes: `planner` → `executor` → `replanner` (on failure) → `aggregator`.
`executor` marks each step `done` or `failed` and returns `Command(goto=...)` accordingly.
`replanner` rewrites only the remaining steps, not the ones already completed.
Cap replanning at 2 cycles. Aggregate all completed results into a final report.
Pick an objective at random from `RESEARCH_OBJECTIVES`.

---

## 07 — Multi-Agent Debate

**File:** `07_multi_agent_debate/exercise.py`

Build 3 nodes: `advocate` → `skeptic` → `arbiter`.
`arbiter` returns `Command(goto="advocate")` to continue or `Command(goto=END)` to deliver a verdict.
Stop when the arbiter issues a VERDICT or after `MAX_TURNS = 6`.
Accumulate all turns in `state["messages"]` using `Annotated[list, operator.add]`.
Pick a topic at random from `DEBATE_TOPICS`.

---

## 08 — Supervisor Hub-and-Spoke

**File:** `08_supervisor_hub_spoke/exercise.py`

Build a `supervisor` node plus 3 worker nodes: `fundamental_analyst`, `quant_analyst`, `risk_officer`.
All worker edges return to `supervisor` — the supervisor reads every result before deciding who's next.
`supervisor` returns `Command(goto=worker)` until all three have reported, then synthesizes a final brief and goes to END.
The `risk_officer` should receive the other two reports as context.
Pick a company at random from `ANALYSIS_SUBJECTS`.

---

## 09 — Parallel Fan-Out → Fan-In

**File:** `09_parallel_fanout_fanin/exercise.py`

Build a `dispatcher` function that returns a list of `Send("worker", {...})` objects — one per domain (web, data, tech, sentiment).
All 4 workers run in parallel via `add_conditional_edges(START, dispatcher, then="aggregator")`.
Worker results accumulate in `state["results"]` using `Annotated[list, operator.add]`.
`aggregator` synthesizes a Competitive Intelligence Report with a threat level rating.
Pick a company at random from `TARGETS`.

---

## 10 — Orchestrator-Worker (Dynamic)

**File:** `10_orchestrator_worker_dynamic/exercise.py`

Build an `orchestrator` node that calls the LLM to decide what subtasks are needed,
then a `dispatcher` function (no LLM call) that converts `state["subtasks"]` into `Send` objects.
Different requests must produce different numbers of subtasks — that's what makes this dynamic.
`aggregator` synthesizes all worker results into a unified response.
Cap subtasks at 6. Pick a request at random from `REQUESTS`.

---

## 11 — Swarm / Handoff

**File:** `11_handoff_swarm/exercise.py`

Build 3 nodes: `generalist`, `billingagent`, `technicalagent`.
Each node responds to the customer, then decides: `RESOLVE` (go to END) or `TRANSFER_TO:<node_name>`.
On transfer, write a handoff note explaining the situation and return `Command(goto=next_node)`.
Context accumulates across handoffs via `Annotated[list, operator.add]`.
Cap at `MAX_TURNS = 8`. Pick a scenario at random from `SCENARIOS`.

---

## 12 — Human-in-the-Loop with Checkpointing

**File:** `12_hitl_checkpoint/exercise.py`

Build 5 nodes: `analyze` → `propose_action` → `await_approval` → `execute_action` → `post_mortem`.
`await_approval` calls `interrupt()` to pause the graph.
Resume by calling `app.invoke(Command(resume=decision), config)` with the same `thread_id`.
Compile the graph with `MemorySaver` as `checkpointer`.
Simulate all three operator paths: `approve`, `modify`, and `reject`.
Pick an incident at random from `INCIDENTS`.

---

## 13 — Supervisor + Parallel ReAct Workers

**File:** `13_supervisor_react_agents/exercise.py`

Build a fan-out/fan-in due-diligence graph: supervisor dispatches 3 parallel workers via `Send("react_worker", ...)`.
Each worker runs a domain-specific `create_react_agent` (`architecture`, `security`, `team`) with its own toolset and returns only a final summary.
After all workers finish, `supervisor_synthesize` combines summaries into a final acquisition recommendation.
Pick a target at random from `ACQUISITION_TARGETS`.

---

## 14 — Sequential Pipeline with ReAct Stages

**File:** `14_sequential_pipeline_react/exercise.py`

Rebuild the same 4-stage linear topology from exercise 01, but each stage is a `create_agent` worker with tools.
Use shared tools `search_archive` and `verify_claim`; each node should invoke only its own agent and pass forward final output.
Keep routing strictly linear: `extractor -> writer -> fact_checker -> publisher -> END`.
Pick one snippet at random from `RAW_RESEARCH_SNIPPETS`.

---

## 15 — Reflection Loop with ReAct Generator

**File:** `15_reflection_react/exercise.py`

Implement the same two-node reflection loop as exercise 03, but `generate` uses `create_agent` with tools.
`critique` should still return `Command(goto="generate" | END)` and parse score with regex.
Use `get_code_pattern` and `check_common_pitfalls` to improve drafts across iterations.
Stop at score >= 8 or after `MAX_ITERATIONS = 4`, using a random task from `CODING_TASKS`.

---

## 16 — Evaluator-Optimizer with ReAct Writer

**File:** `16_evaluator_optimizer_react/exercise.py`

Keep the evaluator-optimizer topology from exercise 04, but replace `generate` with a tool-using `create_agent`.
The evaluator node returns `Command(goto="generate" | END)` and scores copy with the 4-dimension rubric (Clarity, Specificity, SEO, CTA).
Use tools `get_product_examples` and `check_readability_score` to improve each iteration.
Stop when score >= 82 or after 3 iterations; pick a random brief from `PRODUCT_BRIEFS`.

---

## 17 — Router with ReAct Specialists + Follow-Up Check

**File:** `17_router_react/exercise.py`

Build a `router` node that classifies message intent to one department and routes via `Command(goto=...)`.
Department nodes (`billing`, `technical`, `account`, `sales`) must use `create_agent` + tools (`lookup_kb_article`, `find_similar_cases`) instead of static prompts.
After specialist response, run `follow_up_check` that routes to END or a second specialist when the answer includes `ALSO_NEEDS:<department>`.
Use a random user issue from `SUPPORT_MESSAGES`.

---

## 18 — Plan-and-Execute with ReAct Planner/Replanner

**File:** `18_planning_execute_react/exercise.py`

Preserve the exercise 06 control flow: `planner -> executor -> (executor|replanner|aggregator)`.
Implement planner, replanner, and aggregator with `create_agent`; keep executor as explicit control node returning `Command`.
Use tools `research_domain` and `get_planning_framework` to create and revise steps.
Cap replanning at `MAX_REPLAN_CYCLES = 2`; select an objective from `RESEARCH_OBJECTIVES`.

---

## 19 — Debate with ReAct Debaters

**File:** `19_debate_react/exercise.py`

Keep the 3-node debate topology from exercise 07 (`advocate -> skeptic -> arbiter`) and state accumulation in `messages`.
Upgrade advocate and skeptic to `create_agent` with evidence tools (`find_evidence`, `get_real_world_case`).
Arbiter returns `Command(goto="advocate" | END)` and issues a verdict or keeps the debate running.
Use `MAX_TURNS = 6` and choose a random topic from `DEBATE_TOPICS`.

---

## 20 — Sequential Supervisor Hub-and-Spoke with ReAct Workers

**File:** `20_supervisor_hub_spoke_react/exercise.py`

Build a true hub-and-spoke controller where `supervisor` routes one worker at a time and receives each report before next delegation.
Workers (`fundamental_analyst`, `quant_analyst`, `risk_officer`) are `create_agent` nodes with financial tools.
All worker edges return to `supervisor`; stop when all reports are present or `MAX_DELEGATIONS` is reached, then synthesize `final_brief`.
Pick a company from `ANALYSIS_SUBJECTS`.

---

## 21 — Parallel Fan-Out/Fan-In with ReAct Workers

**File:** `21_parallel_fanout_react/exercise.py`

Preserve the fan-out/fan-in topology from exercise 09: dispatcher returns multiple `Send` objects and workers run in parallel.
Upgrade worker and aggregator nodes to `create_agent`; dispatcher remains a plain function.
Use tool-backed intelligence (`get_competitive_data`, `get_market_benchmarks`) so each domain report is grounded in retrieved data.
Aggregate worker outputs from `state["results"]` (`Annotated[list, operator.add]`) into a final competitive report.

---

## 22 — Dynamic Orchestrator-Worker with ReAct + Plan Validation

**File:** `22_orchestrator_worker_react/exercise.py`

Keep dynamic decomposition from exercise 10, but use `create_agent` for orchestrator, worker, and aggregator nodes.
Insert a `plan_validator` control node after orchestrator that returns `Command(goto="dispatcher" | "orchestrator")`.
Allow at most one replan cycle (`MAX_REPLAN_CYCLES = 1`) before dispatching subtasks.
Dispatcher remains non-LLM: convert `state["subtasks"]` into `Send("worker", ...)`, cap subtasks at `MAX_SUBTASKS = 6`.

---

## 23 — Human-in-the-Loop with ReAct Agents

**File:** `23_hitl_react/exercise.py`

Keep exercise 12 topology and checkpoint flow, but upgrade `analyze`, `propose_action`, `execute_action`, and `post_mortem` to `create_agent`.
Do not replace `await_operator_approval`: it must use `interrupt()` and route with `Command(goto="execute_action" | "abort")`.
Compile with `MemorySaver`, run phase 1 until interrupt, then resume with `Command(resume=decision)` using the same `thread_id`.
Use tool-backed incident context (`run_service_diagnostic`, `check_runbook_exists`, `get_historical_incidents`) and test decisions `approve`, `modify`, and `reject`.

---

## Setup

```bash
cd multi-agentic-exercises
pip install langgraph langchain-openai langchain-core langfuse python-dotenv
cp .env .env.local   # fill in OPENAI_API_KEY and LANGFUSE_* keys
python 01_sequential_pipeline/exercise.py
```
