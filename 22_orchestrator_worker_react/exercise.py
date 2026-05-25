"""
Exercise 22 — Dynamic Orchestrator-Worker with ReAct + Plan Validation
========================================================================
Task:
    Build a dynamic orchestrator-worker graph with a plan quality gate:
      orchestrator -> plan_validator -> dispatcher -> worker(s) -> aggregator

    Orchestrator, worker, and aggregator must be create_agent nodes using:
      - research_topic_depth(topic)
      - get_domain_examples(domain)

    plan_validator is a Command-returning control node:
      - goto="dispatcher" when plan is coherent
      - goto="orchestrator" for one replan cycle when plan is weak

    dispatcher must be non-LLM and convert subtasks to Send objects.
    Cap at MAX_SUBTASKS = 6 and MAX_REPLAN_CYCLES = 1.
    Pick one request at random from REQUESTS.

Libraries:
    langgraph — StateGraph, START, END
    langgraph.types — Send, Command
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage
    typing — TypedDict, Annotated, Optional
    operator
"""

import operator
import random
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

MAX_SUBTASKS = 6
MAX_REPLAN_CYCLES = 1

REQUESTS = [
    {"title": "Launch readiness review", "domain": "product launch"},
    {"title": "Post-incident root cause analysis", "domain": "incident analysis"},
    {"title": "Engineering team scaling plan", "domain": "engineering hiring"},
    {"title": "AI cost optimization strategy", "domain": "llm cost"},
    {"title": "Competitive positioning response", "domain": "competitive strategy"},
]


class WorkerState(TypedDict):
    subtask: str
    subtask_index: int
    domain: str
    result: str


class OrchestratorState(TypedDict):
    request: str
    domain: str
    subtasks: list[str]
    results: Annotated[list, operator.add]
    final_synthesis: str
    replan_count: int
    validation_feedback: Optional[str]
