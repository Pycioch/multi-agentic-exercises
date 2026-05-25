"""
Exercise 17 — Router with ReAct Specialists + Follow-Up Check
==============================================================
Task:
    Build a routing graph for support requests:
      router -> specialist -> follow_up_check -> (another specialist | END)

    Router classifies message into one department:
      billing, technical, account, sales

    Each specialist must be a create_agent node using tools:
      - lookup_kb_article(department, issue_type)
      - find_similar_cases(issue_description)

    follow_up_check reads specialist output. If it contains
    'ALSO_NEEDS:<department>' and the department is valid, route there.
    Otherwise route to END.

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage, SystemMessage
"""

import random
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

VALID_DEPARTMENTS = ["billing", "technical", "account", "sales"]

SUPPORT_MESSAGES = [
    "I was charged twice for my subscription this month. I need a refund ASAP.",
    "My API calls return 429 errors even though traffic is below my limit.",
    "I forgot my password and the reset email is not arriving.",
    "Can I add a second admin to my account for my CTO?",
    "The webhook for order.completed stopped firing after your maintenance window.",
    "Is Enterprise cheaper than Pro for a team of 15 seats?",
]


class RouterState(TypedDict):
    message: str
    department: str
    routing_reason: str
    response: str
    follow_up_department: str
