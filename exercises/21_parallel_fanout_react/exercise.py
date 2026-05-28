"""
Exercise 21 — Parallel Fan-Out/Fan-In with ReAct Workers
=========================================================
Task:
    Preserve the fan-out/fan-in topology from exercise 09:
      - dispatcher returns list[Send("worker", ...)]
      - workers run in parallel
      - aggregator synthesizes all worker outputs

    Upgrade worker and aggregator to create_agent nodes using:
      - get_competitive_data(company, domain)
      - get_market_benchmarks(sector)

    Worker results accumulate in state["results"] with Annotated[list, operator.add].
    Pick one target company at random from TARGETS.

Libraries:
    langgraph — StateGraph, START, END
    langgraph.types — Send
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage
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

TARGETS = [
    {"company": "Linear", "sector": "project management"},
    {"company": "Retool", "sector": "low-code"},
    {"company": "Supabase", "sector": "backend platform"},
    {"company": "Warp Terminal", "sector": "developer tools"},
    {"company": "Descript", "sector": "video editing"},
]


class WorkerState(TypedDict):
    company: str
    description: str
    sector: str
    domain: str
    result: str


class IntelState(TypedDict):
    target: dict
    results: Annotated[list, operator.add]
    final_report: str
