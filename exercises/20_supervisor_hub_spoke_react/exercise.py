"""
Exercise 20 — Sequential Supervisor Hub-and-Spoke with ReAct Workers
=====================================================================
Task:
    Build a true hub-and-spoke controller:
      supervisor -> worker -> supervisor -> ... -> END

    Required worker nodes:
      - fundamental_analyst
      - quant_analyst
      - risk_officer

    Each worker is a create_agent with financial tools.
    The supervisor is a Command-returning control node that decides the next worker
    after reading current reports. All worker edges return to supervisor.

    Stop when all three reports are filled or MAX_DELEGATIONS is reached.
    Then synthesize final_brief and route to END.

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
    langchain.agents — create_agent
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

MAX_DELEGATIONS = 8

ANALYSIS_SUBJECTS = [
    {
        "company": "Stripe",
        "sector": "FinTech / Payments Infrastructure",
        "context": "Private valuation reset, large payment volume, strong competition and regulatory pressure.",
    },
    {
        "company": "Snowflake",
        "sector": "Cloud Data Warehousing",
        "context": "Strong revenue base with decelerating growth and AI strategy transition.",
    },
    {
        "company": "Klarna",
        "sector": "BNPL / Consumer Credit",
        "context": "Profitable year before IPO ambition with meaningful consumer-credit regulatory exposure.",
    },
    {
        "company": "Palantir",
        "sector": "AI / Government & Enterprise Analytics",
        "context": "High valuation multiple with strong US commercial momentum and concentration risk.",
    },
    {
        "company": "Mistral AI",
        "sector": "Foundation Model Provider",
        "context": "High-growth European foundation model company with pricing and commoditization pressure.",
    },
]


class AnalysisState(TypedDict):
    subject: dict
    messages: Annotated[list, operator.add]
    fundamental_report: str
    quant_report: str
    risk_report: str
    final_brief: str
    delegation_count: int
