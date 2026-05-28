"""
Exercise 18 — Plan-and-Execute with ReAct Planner/Replanner
=============================================================
Task:
    Preserve the plan-execute-replan topology from exercise 06:
      planner -> executor -> (executor | replanner | aggregator)

    planner, replanner, and aggregator must be create_agent nodes using:
      - research_domain(topic)
      - get_planning_framework(objective_type)

    executor remains an explicit control node and returns Command routing.
    Replanning is capped at MAX_REPLAN_CYCLES = 2.
    Pick an objective at random from RESEARCH_OBJECTIVES.

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage
    typing — TypedDict, Optional
"""

import random
from typing import Optional, TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

MAX_REPLAN_CYCLES = 2

RESEARCH_OBJECTIVES = [
    "Compare top LLM API providers for 1M requests/month production workload.",
    "Design a multilingual legal RAG architecture with clause-level citations.",
    "Evaluate build-vs-buy options for an internal developer platform.",
    "Compare Kubernetes vs serverless for bursty ML inference traffic.",
]


class Step(TypedDict):
    description: str
    status: str
    result: Optional[str]


class PlanState(TypedDict):
    objective: str
    steps: list
    current_index: int
    replan_count: int
    final_report: str
