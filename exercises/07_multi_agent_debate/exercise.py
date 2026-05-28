"""
Exercise 07 — Multi-Agent Debate
===================================
Task:
    Build an adversarial three-agent debate. Two agents argue opposing sides of a
    technology decision; a third evaluates and eventually delivers a verdict.
    Pick one topic at random from DEBATE_TOPICS below.

    Nodes to build:
      - advocate — builds the strongest case FOR the given position;
                   addresses the Skeptic's latest objection directly
      - skeptic  — challenges every claim, demands evidence, finds failure modes;
                   must not accept vague assertions
      - arbiter  — after each exchange evaluates if the debate has produced enough
                   clarity to deliver a verdict; returns CONTINUE or a structured VERDICT

    Routing: use Command(goto="advocate" | END) inside arbiter.
    Stop when the arbiter issues a VERDICT or after MAX_TURNS = 6.
    Accumulate all turns in state["messages"] using Annotated[list, operator.add].

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
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

DEBATE_TOPICS = [
    {
        "question": "Unit tests are a waste of time — integration tests and type hints are enough.",
        "advocate_position": (
            "Yes — unit tests mock away all the interesting behaviour and give false confidence. "
            "They break on every refactor, slow CI, and test the wrong thing: isolated functions "
            "that never fail in isolation. A strong integration test suite plus strict typing "
            "catches real bugs. Delete the mocks, ship faster."
        ),
    },
    {
        "question": "Code review is net-negative for team velocity and should be replaced by pair programming.",
        "advocate_position": (
            "Yes — async code review creates multi-day feedback loops, breeds passive-aggressive "
            "comment threads, and lets reviewers nitpick style instead of catching logic errors. "
            "Pair programming delivers the same quality gate in real time, spreads knowledge "
            "continuously, and costs the same engineer-hours with far less context-switching overhead."
        ),
    },
    {
        "question": "Microservices were a mistake — the industry should move back to modular monoliths.",
        "advocate_position": (
            "Yes — distributed systems are hard, and most teams adopted microservices to solve "
            "org problems they should have fixed with better abstractions. Network latency, "
            "distributed transactions, and 12 Kubernetes clusters to run a TODO app are not "
            "engineering wins. A well-structured monolith with clear module boundaries scales "
            "further than most companies will ever need."
        ),
    },
    {
        "question": "Sprints and story points are cargo-cult management that should be abolished.",
        "advocate_position": (
            "Yes — two-week sprints force artificial deadlines on work that doesn't fit, "
            "and story points are a pseudo-metric that measures nothing except how good "
            "the team is at estimating story points. Continuous flow with explicit WIP limits "
            "and cycle time measurement gives better predictability with zero planning ceremony overhead."
        ),
    },
    {
        "question": "Every production service should be owned by a dedicated on-call engineer, not rotated across the whole team.",
        "advocate_position": (
            "Yes — rotation-based on-call spreads context so thin that engineers page each other "
            "for incidents they have never seen before. Dedicated ownership creates accountability, "
            "deep system knowledge, and the right incentive to fix root causes instead of silencing "
            "alerts. Fear of being on-call forever is what prevents bad architecture from being built."
        ),
    },
]
