"""LangGraph Pipeline: planner → extractor → visualizer.

This is the first explicit StateGraph in the progression. Prior steps used
create_agent() which builds a graph internally; here we build the graph ourselves
so its structure is visible, testable, and extensible.

The pipeline handles questions that decompose naturally into sequential stages:
  1. Understand and plan (planner)
  2. Retrieve evidence from CSVs (extractor)
  3. Render a chart if requested (visualizer)

For simpler questions (single lookup, no chart), the CLI routes to the
ReAct agent from step 4 instead. See cli/run.py for the routing heuristic.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from cloudops.core.budget import BudgetConfig, DEFAULT_BUDGET
from cloudops.graph.nodes import extractor_node, planner_node, visualizer_node
from cloudops.observability.opik import build_langchain_config


class PipelineState(TypedDict):
    """State carried through the planner → extractor → visualizer pipeline.

    messages:          conversation history (add_messages reducer appends, never replaces)
    plan:              JSON string produced by the planner node
    extraction_result: text summary produced by the extractor node
    tokens_used:       running token count; checked against budget at each node
    budget:            immutable limits for this run (max_tokens, recursion_limit)
    _final_answer:     set by visualizer; used by CLI to extract the response
    """

    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    extraction_result: str
    tokens_used: int
    budget: BudgetConfig
    _final_answer: str


def build_pipeline(budget: BudgetConfig = DEFAULT_BUDGET):
    """Return a compiled Pipeline graph with MemorySaver checkpointing.

    The recursion_limit from budget is passed to RunnableConfig at invoke time
    (see cli/run.py) — it caps how many node transitions LangGraph will allow
    before raising GraphRecursionError.
    """
    builder = StateGraph(PipelineState)

    builder.add_node("planner", planner_node)
    builder.add_node("extractor", extractor_node)
    builder.add_node("visualizer", visualizer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "extractor")
    builder.add_edge("extractor", "visualizer")
    builder.add_edge("visualizer", END)

    return builder.compile(checkpointer=MemorySaver())


def run_pipeline(
    pipeline,
    question: str,
    thread_id: str,
    budget: BudgetConfig = DEFAULT_BUDGET,
    session_id: str | None = None,
    trace_name: str = "cloudops-step-07-pipeline",
) -> str:
    """Invoke the pipeline for one question and return the final answer string."""
    from langchain_core.messages import HumanMessage

    config = build_langchain_config(
        run_name=trace_name,
        session_id=session_id,
        tags=["cloudops-step-07", "runtime", "pipeline"],
        metadata={"entrypoint": "cloudops"},
        configurable={"thread_id": thread_id},
        recursion_limit=budget.recursion_limit,
    )
    initial_state: dict[str, Any] = {
        "messages": [HumanMessage(content=question)],
        "plan": "",
        "extraction_result": "",
        "tokens_used": 0,
        "budget": budget,
        "_final_answer": "",
    }
    result = pipeline.invoke(initial_state, config=config)
    return result.get("_final_answer") or result.get("extraction_result") or "(no response)"
