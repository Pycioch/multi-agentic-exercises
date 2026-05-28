"""
Exercise 16 — Evaluator-Optimizer with ReAct Writer
====================================================
Task:
    Build an evaluator-optimizer loop where the writer is a create_agent and
    evaluator returns Command(goto="generate" | END).

    Required tools:
      - get_product_examples(category)
      - check_readability_score(text)

    Stop when total score >= 82/100 or after MAX_ITERATIONS = 3.
    Pick a random brief from PRODUCT_BRIEFS below.

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage
    re
"""

import random
from typing import Optional, TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

MAX_ITERATIONS = 3
SCORE_THRESHOLD = 82

PRODUCT_BRIEFS = [
    {
        "product": "DataSync Pro",
        "category": "B2B SaaS / data integration",
        "audience": "CTOs and data engineers at mid-market companies",
        "primary_benefit": "Eliminate manual ETL bottlenecks",
    },
    {
        "product": "FocusBrew Coffee Subscription",
        "category": "D2C / food & beverage",
        "audience": "Remote workers and freelancers aged 25-40",
        "primary_benefit": "Sustained focus without the afternoon crash",
    },
    {
        "product": "LegalDraft AI",
        "category": "Legal tech / AI assistant",
        "audience": "Solo practitioners and small law firms",
        "primary_benefit": "Cut contract drafting time by 70%",
    },
    {
        "product": "TrailGuard GPS Watch",
        "category": "Consumer hardware / outdoor fitness",
        "audience": "Serious hikers and ultra-marathon runners",
        "primary_benefit": "Never get lost, even off the grid",
    },
]

EVALUATOR_RUBRIC = """\
Score on 4 dimensions (25 pts each, total 100):
1. CLARITY
2. SPECIFICITY
3. SEO & KEYWORDS
4. CALL-TO-ACTION
End with: Total: XX/100
"""


class EvalState(TypedDict):
    brief: dict
    description: str
    evaluation: str
    total_score: int
    iteration: int
    feedback_summary: Optional[str]
