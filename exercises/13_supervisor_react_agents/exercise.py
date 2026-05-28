"""
Exercise 13 — Supervisor + Parallel ReAct Workers
==================================================
Task:
    Build a due-diligence graph with one supervisor and three autonomous workers.
    The supervisor should dispatch all workers in parallel with Send API:
      - architecture
      - security
      - team

    Each worker is a create_react_agent loop with domain-specific tools.
    Workers return only final summaries to supervisor (not internal tool traces).
    After all three workers finish, synthesize a final recommendation.

    Pick one target at random from ACQUISITION_TARGETS below.

Libraries:
    langgraph — StateGraph, START, END
    langgraph.types — Send
    langgraph.prebuilt — create_react_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage, SystemMessage
    typing — TypedDict, Annotated
    operator
"""

import operator
import random
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

ACQUISITION_TARGETS = [
    {
        "company": "DataFlow Systems",
        "sector": "Data Pipeline / ETL",
        "size": "80 engineers, Series B, ~$18M ARR",
        "context": (
            "Real-time data pipelines for e-commerce on AWS. "
            "Strong growth but rising SMB churn and a CTO bottleneck in architecture decisions."
        ),
    },
    {
        "company": "SecureVault Inc.",
        "sector": "Identity & Access Management",
        "size": "45 engineers, Series A, ~$8M ARR",
        "context": (
            "Enterprise IAM with heavy financial-sector exposure. "
            "ISO 27001 certified but mid-migration from monolith to microservices."
        ),
    },
    {
        "company": "DevMetrics AI",
        "sector": "Engineering Analytics",
        "size": "30 engineers, Seed+, ~$3M ARR",
        "context": (
            "Fast-moving analytics startup with strong product velocity. "
            "No dedicated security function and shared cloud credentials."
        ),
    },
    {
        "company": "CloudNative Solutions",
        "sector": "Kubernetes / Platform Engineering",
        "size": "120 engineers, Series C, ~$35M ARR",
        "context": (
            "Enterprise internal developer platform with strong operational maturity. "
            "Complex multi-cloud footprint and high enterprise expectations."
        ),
    },
    {
        "company": "AISearch Corp",
        "sector": "AI-Native Search",
        "size": "55 engineers, Series B, ~$12M ARR",
        "context": (
            "Vector search provider with strong ML talent. "
            "Recent parser security incidents and weak distributed systems depth."
        ),
    },
]


class WorkerState(TypedDict):
    company: str
    context: str
    domain: str
    summary: str


class SupervisorState(TypedDict):
    target: dict
    summaries: Annotated[list, operator.add]
    final_report: str
