"""
Exercise 15 — Reflection Loop with ReAct Generator
===================================================
Task:
    Build a 2-node reflection loop:
      generate -> critique

    The generate node must use create_agent with tools:
      - get_code_pattern
      - check_common_pitfalls

    The critique node returns Command(goto="generate" | END) based on score.
    Parse score via regex. Stop when score >= 8 or after MAX_ITERATIONS = 4.
    Pick one task from CODING_TASKS at random.

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage
    re
"""

import random
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

MAX_ITERATIONS = 4
SCORE_THRESHOLD = 8

CODING_TASKS = [
    "Write a thread-safe Singleton class in Python with context manager support.",
    "Implement a retry decorator with exponential backoff and jitter.",
    "Implement a generic O(1) LRU cache class with get() and put().",
    "Write an async rate limiter that allows N requests per second.",
    "Implement a simple event sourcing aggregate with replay support.",
]


class ReflectionState(TypedDict):
    task: str
    draft: str
    critique: str
    score: int
    iteration: int
