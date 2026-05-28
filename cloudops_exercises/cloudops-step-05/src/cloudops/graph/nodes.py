"""Pipeline node functions: planner → extractor → visualizer.

Each node has one job:
- planner:    decompose the user question into a structured plan (no tool calls)
- extractor:  execute data retrieval using CSV/pandas tools (one round of tool calls)
- visualizer: render a chart if requested, or pass through the data summary

Token usage from each node is accumulated in state["tokens_used"] and checked
against the budget before the node returns.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from cloudops.core.budget import BudgetConfig, check_budget
from cloudops.core.routing import get_model
from cloudops.prompts.personas import EXTRACTOR_PROMPT, PLANNER_PROMPT, VISUALIZER_PROMPT
from cloudops.tools.csv_tools import describe_csv, list_csv_files, query_csv
from cloudops.tools.pandas_tools import (
    aggregate_csv,
    compute_correlation,
    timeseries_resample,
    top_n,
)
from cloudops.tools.viz_tools import plot_bar, plot_histogram, plot_timeseries

EXTRACTOR_TOOLS = [
    list_csv_files,
    describe_csv,
    query_csv,
    aggregate_csv,
    timeseries_resample,
    compute_correlation,
    top_n,
]

VIZ_TOOLS = [plot_timeseries, plot_bar, plot_histogram]

_EXTRACTOR_TOOL_MAP = {t.name: t for t in EXTRACTOR_TOOLS}
_VIZ_TOOL_MAP = {t.name: t for t in VIZ_TOOLS}


def _execute_tool_calls(response, tool_map: dict) -> tuple[list[ToolMessage], str]:
    """Run any tool calls in the LLM response.

    Returns a tuple of (tool_messages, combined_text).
    - tool_messages: ToolMessage objects with correct tool_call_ids (required by OpenAI API)
    - combined_text: human-readable concatenation of all results

    This is a single-round tool execution — the LLM makes one decision,
    we run the tools, and return. Looping is the graph's job, not the node's.
    """
    if not response.tool_calls:
        return [], response.content or "(no output)"

    tool_messages: list[ToolMessage] = []
    parts: list[str] = []

    for tc in response.tool_calls:
        fn = tool_map.get(tc["name"])
        if fn is None:
            result_text = f"unknown tool: {tc['name']}"
        else:
            try:
                result_text = str(fn.invoke(tc["args"]))
            except Exception as exc:
                result_text = f"ERROR: {exc}"

        tool_messages.append(
            ToolMessage(content=result_text, tool_call_id=tc["id"])
        )
        parts.append(f"[tool: {tc['name']}]\n{result_text}")

    return tool_messages, "\n\n".join(parts)


def planner_node(state: dict) -> dict:
    """Decompose the user question into {extraction_query, viz_request, reasoning}."""
    budget: BudgetConfig = state["budget"]
    llm = get_model("plan")

    messages = [SystemMessage(content=PLANNER_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)

    tokens = response.usage_metadata.get("total_tokens", 0)
    check_budget(state["tokens_used"], tokens, budget)

    # Parse the JSON plan; fall back to raw text so the pipeline never hard-crashes.
    try:
        plan = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        plan = {"extraction_query": response.content, "viz_request": None, "reasoning": ""}

    return {
        "plan": json.dumps(plan),
        "messages": [response],
        "tokens_used": state["tokens_used"] + tokens,
    }


def extractor_node(state: dict) -> dict:
    """Execute the extraction_query from the plan using data tools."""
    budget: BudgetConfig = state["budget"]
    llm = get_model("extract").bind_tools(EXTRACTOR_TOOLS)

    plan = json.loads(state["plan"])
    extraction_query = plan.get("extraction_query", state["plan"])

    messages = [
        SystemMessage(content=EXTRACTOR_PROMPT),
        HumanMessage(content=extraction_query),
    ]
    response = llm.invoke(messages)

    tokens = response.usage_metadata.get("total_tokens", 0)
    check_budget(state["tokens_used"], tokens, budget)

    # Execute whatever tools the LLM decided to call.
    tool_msgs, tool_text = _execute_tool_calls(response, _EXTRACTOR_TOOL_MAP)

    # If there were tool calls, do a follow-up LLM call to produce a text summary.
    # OpenAI requires ToolMessage objects (not HumanMessage) after an assistant tool_call.
    if response.tool_calls:
        followup_messages = messages + [response] + tool_msgs
        summary_response = llm.invoke(followup_messages)
        summary_tokens = summary_response.usage_metadata.get("total_tokens", 0)
        check_budget(state["tokens_used"] + tokens, summary_tokens, budget)
        tokens += summary_tokens
        extraction_result = summary_response.content or tool_text
    else:
        extraction_result = tool_text

    return {
        "extraction_result": extraction_result,
        "messages": [response],
        "tokens_used": state["tokens_used"] + tokens,
    }


def visualizer_node(state: dict) -> dict:
    """Render a chart if viz_request is present; otherwise pass through the data summary."""
    budget: BudgetConfig = state["budget"]

    plan = json.loads(state["plan"])
    viz_request = plan.get("viz_request")

    if not viz_request:
        # Nothing to visualize — return the extraction result directly.
        return {
            "messages": [HumanMessage(content=state["extraction_result"])],
            "tokens_used": state["tokens_used"],
        }

    llm = get_model("visualize").bind_tools(VIZ_TOOLS)

    prompt = f"{state['extraction_result']}\n\nViz request: {viz_request}"
    messages = [
        SystemMessage(content=VISUALIZER_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)

    tokens = response.usage_metadata.get("total_tokens", 0)
    check_budget(state["tokens_used"], tokens, budget)

    _, chart_output = _execute_tool_calls(response, _VIZ_TOOL_MAP)

    final_text = f"{state['extraction_result']}\n\n{chart_output}" if chart_output != response.content else response.content

    return {
        "messages": [response],
        "tokens_used": state["tokens_used"] + tokens,
        "_final_answer": final_text,
    }
