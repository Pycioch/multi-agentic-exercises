"""
Exercise 01 — Sequential Pipeline
===================================
Task:
    Process a raw research snippet through a 4-step linear pipeline.
    Pick one snippet at random from RAW_RESEARCH_SNIPPETS below and pass it
    through four agents in order:
      1. Extractor   — pull the 3 key facts from the raw text
      2. Writer      — turn those facts into a polished newsletter paragraph
      3. Fact-Checker — flag any dubious or unverifiable claims
      4. Publisher   — produce a tweet-length summary (≤280 chars)

    Each agent only reads the output of the previous one. No branching, no loops.
    Print each agent's output as the pipeline runs.

Libraries:
    langgraph  — StateGraph, END, add_edge
    langchain_openai — ChatOpenAI
    langchain_core.messages — HumanMessage
    langfuse.langchain — CallbackHandler  (trace every LLM call)
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

RAW_RESEARCH_SNIPPETS = [
    """\
    Klarna's AI assistant handled 2.3 million customer conversations in its first month,
    doing the work of 700 full-time agents. Average resolution time dropped from
    11 minutes to 2 minutes. Customer satisfaction held steady at pre-AI levels.
    The project cost roughly $3M to deploy and saved an estimated $40M annually.
    The system occasionally misrouted refund requests to the wrong department.
    """,
    """\
    A 2025 study by Stanford HAI found that GPT-4-class models hallucinate on
    medical questions roughly 14% of the time even when retrieval-augmented.
    Fine-tuned models on PubMed cut that to 6%. Latency for RAG-augmented
    responses was 2.1 seconds on average. Smaller 7B models hallucinated at 31%.
    The study covered 10,000 questions across cardiology, oncology, and neurology.
    """,
    """\
    LangGraph surpassed CrewAI in GitHub stars in early 2026, reaching 95k stars
    after its v1.0 stable release. Enterprise adoption doubled year-over-year,
    with Uber, LinkedIn, and Klarna all running LangGraph in production.
    The framework's built-in checkpointing with PostgreSQL handles 40k state
    persists per second in load tests. Its TypeScript port reached feature parity
    in Q4 2025 and now accounts for 30% of weekly downloads.
    """,
    """\
    OpenAI's o3 model scored 87.5% on the ARC-AGI benchmark, compared to 4%
    for GPT-4o. The model uses extended compute at inference time, averaging
    $15 per problem solved. It achieved bronze-medal performance on the
    International Mathematical Olympiad. Training cost was not disclosed
    but estimated at over $100M. The model is available via API with a
    $200/month minimum commitment.
    """,
]
