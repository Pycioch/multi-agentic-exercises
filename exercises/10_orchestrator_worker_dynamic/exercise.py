"""
Exercise 10 — Orchestrator-Worker (Dynamic Decomposition)
============================================================
Task:
    Build an orchestrator that REASONS about how to split a complex request into
    parallel subtasks at runtime, fires workers for each, then aggregates the results.
    Pick one request at random from REQUESTS below.
    Different requests should produce different numbers and types of subtasks.

    Nodes to build:
      - orchestrator — reads the request and produces a numbered list of 3-6
                       independent subtasks (each one completable without the others)
      - dispatcher   — reads state["subtasks"] and returns a list of Send objects
                       (one per subtask); this is a pure routing function, not an LLM call
      - worker       — executes one subtask; produces specific, concrete output
      - aggregator   — synthesizes all worker results into a unified response

    Routing: add_conditional_edges("orchestrator", dispatcher, then="aggregator").
    Cap subtasks at 6. Parse orchestrator output with regex (numbered lines only).

Libraries:
    langgraph — StateGraph, END, START
    langgraph.types — Send
    langchain_openai — ChatOpenAI
    langchain_core.messages — HumanMessage, SystemMessage
    langfuse.langchain — CallbackHandler
    typing — Annotated
    operator
"""

import random, os, uuid, operator
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

REQUESTS = [
    {
        "title": "Launch readiness review for a new B2B SaaS product",
        "request": (
            "We're launching DataSync Pro (B2B data integration tool) in 6 weeks. "
            "We need: market positioning, pricing validation, technical reliability checklist, "
            "go-to-market timeline, and risk identification."
        ),
    },
    {
        "title": "Post-incident root cause analysis",
        "request": (
            "Our API gateway had a 45-minute outage yesterday (09:15–10:00 UTC). "
            "Symptoms: 18% error rate, p99 latency 8s, circuit breakers triggered. "
            "Identify root cause, contributing factors, immediate fixes, and systemic changes."
        ),
    },
    {
        "title": "Engineering team scaling plan",
        "request": (
            "We need to scale our engineering team from 15 to 40 people over 12 months. "
            "Build a hiring plan: role prioritization, compensation bands, interview process, "
            "onboarding program, and how the team structure should evolve."
        ),
    },
    {
        "title": "AI cost optimization strategy",
        "request": (
            "Our LLM API costs doubled last month to $85,000. We use GPT-4o for everything. "
            "Design a cost optimization strategy targeting 60% reduction "
            "without degrading user-facing quality."
        ),
    },
    {
        "title": "Competitive response to a new market entrant",
        "request": (
            "A well-funded startup just launched a direct competitor at 40% lower price, "
            "with $50M Series B and 6-month free trials. "
            "Develop our competitive response strategy across product, pricing, and sales."
        ),
    },
]
