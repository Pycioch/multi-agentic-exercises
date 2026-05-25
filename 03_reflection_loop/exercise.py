"""
Exercise 03 — Reflection / Self-Critique Loop
===============================================
Task:
    Build a two-node loop where a generator writes Python code and a critic
    scores it. The loop runs until the score reaches 8/10 or 4 iterations pass.
    Pick one task at random from CODING_TASKS below.

    Nodes to build:
      - generate — calls LLM to write (or rewrite based on prior critique) Python code
      - critique — calls LLM to review the draft and return a score 1-10 + specific issues

    Routing: use Command(goto="generate" | END) inside critique.
    The critic decides whether to loop back or stop.
    Parse the score with a regex (look for "Score: N/10" or "N/10").

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
    langchain_openai — ChatOpenAI
    langchain_core.messages — HumanMessage
    langfuse.langchain — CallbackHandler
"""

import random, os, uuid
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
SESSION_ID = str(uuid.uuid4())
Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
         secret_key=os.environ["LANGFUSE_SECRET_KEY"],
         host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"))
lf = CallbackHandler(public_key=os.environ["LANGFUSE_PUBLIC_KEY"])

# ── Data ──────────────────────────────────────────────────────────────────────

CODING_TASKS = [
    {
        "title": "Thread-safe singleton in Python",
        "prompt": (
            "Write a thread-safe Singleton class in Python that also supports "
            "being used as a context manager (with-statement). "
            "Include brief inline comments explaining the thread-safety mechanism."
        ),
    },
    {
        "title": "Retry decorator with exponential backoff",
        "prompt": (
            "Write a Python decorator @retry(max_attempts=3, base_delay=0.5) "
            "that retries a function on exception with exponential backoff and jitter. "
            "The decorator should log each retry attempt and re-raise after exhausting retries."
        ),
    },
    {
        "title": "In-memory LRU cache",
        "prompt": (
            "Implement a generic LRU cache in Python as a class with get() and put() methods "
            "and a configurable max_size. The cache should be O(1) for both operations. "
            "Include a usage example in a __main__ block."
        ),
    },
    {
        "title": "Async rate limiter",
        "prompt": (
            "Write an async Python rate limiter that allows at most N requests per second. "
            "It should work with asyncio and expose an async context manager interface. "
            "Handle burst traffic gracefully."
        ),
    },
    {
        "title": "Event sourcing aggregate",
        "prompt": (
            "Implement a simple Event Sourcing pattern in Python: an Aggregate base class "
            "that applies events to build its state, stores an event log, and can replay "
            "events from a given snapshot. Show a concrete BankAccount example."
        ),
    },
]
