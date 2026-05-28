"""
Exercise 09 — Parallel Fan-Out → Fan-In
==========================================
Task:
    Build a competitive intelligence tool that fires 4 independent research agents
    simultaneously, then aggregates their findings into one report.
    Pick one company at random from TARGETS below.

    Nodes to build (all run in parallel):
      - worker "web"       — public web presence, content strategy, marketing positioning
      - worker "data"      — funding history, valuation, headcount signals, revenue estimates
      - worker "tech"      — inferred tech stack, GitHub presence, job posting signals
      - worker "sentiment" — community reception, reviews, social sentiment score

    After all 4 workers finish, one aggregator synthesizes a Competitive Intelligence Report
    with a "Competitive Threat Level" rating: Low / Medium / High / Critical.

    Routing: use add_conditional_edges(START, dispatcher, then="aggregator").
    dispatcher returns a list of Send("worker", {...}) objects — one per domain.
    Worker results accumulate via Annotated[list, operator.add] on state["results"].

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

TARGETS = [
    {
        "company": "Linear",
        "description": (
            "Project management tool targeting software teams. Known for extreme performance "
            "and keyboard-first UX. Backed by Sequoia. ~$35M ARR, ~120 employees (est)."
        ),
    },
    {
        "company": "Retool",
        "description": (
            "Low-code internal tools builder for developers. Raised $45M Series C, ~$1B valuation. "
            "Competes with Appsmith, Tooljet (open source)."
        ),
    },
    {
        "company": "Supabase",
        "description": (
            "Open-source Firebase alternative (PostgreSQL backend). $200M Series D at $2B valuation. "
            "Strong developer community, 70k GitHub stars."
        ),
    },
    {
        "company": "Warp Terminal",
        "description": (
            "AI-powered terminal for developers. Raised $73M total. Mac-first, expanding to Linux/Windows. "
            "4.8/5 on App Store, 500k+ downloads."
        ),
    },
    {
        "company": "Descript",
        "description": (
            "AI video/podcast editor. Raised $100M+. Text-based editing, voice cloning, screen recording. "
            "Competes with Adobe Premiere, CapCut, Riverside.fm."
        ),
    },
]
