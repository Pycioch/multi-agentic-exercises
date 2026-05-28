"""
Exercise 06 — Plan-and-Execute with Replanning
================================================
Task:
    Build a research pipeline that first creates a full plan, then executes it
    step by step. If a step fails, a replanner rewrites the remaining steps.
    At the end, an aggregator synthesizes all results into a report.
    Pick one objective at random from RESEARCH_OBJECTIVES below.

    Nodes to build:
      - planner    — decomposes the objective into 4-5 numbered steps
      - executor   — executes the current step using prior steps as context;
                     marks the step "done" or "failed"
      - replanner  — triggered only on failure; rewrites remaining steps
                     given what has been completed so far
      - aggregator — synthesizes all completed step results into a final report

    Routing: use Command(goto=...) from executor to pick: executor / replanner / aggregator.
    Parse planner output with regex (numbered lines only).
    Cap replanning at 2 cycles.

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

RESEARCH_OBJECTIVES = [
    {
        "title": "LLM API provider comparison for production workloads",
        "objective": (
            "Compare the top 4 LLM API providers (OpenAI, Anthropic, Google, Mistral) "
            "for a production multi-agent system handling 1M requests/month. "
            "Evaluate: pricing at scale, rate limits, model quality on coding tasks, "
            "latency p99, and enterprise support SLAs. Recommend one primary and one fallback."
        ),
    },
    {
        "title": "RAG architecture selection for a legal document system",
        "objective": (
            "Design a RAG (Retrieval-Augmented Generation) architecture for a legal tech "
            "startup processing contracts in English, German, and Polish. "
            "The system must cite clause-level sources, handle 500-page documents, "
            "and achieve <2s retrieval latency. Compare dense vs sparse vs hybrid retrieval."
        ),
    },
    {
        "title": "Build vs buy decision: internal developer platform",
        "objective": (
            "Evaluate whether a 40-person engineering team should build an internal "
            "developer platform (IDP) or adopt Backstage, Port, or Cortex. "
            "Consider: current stack (AWS, GitHub, Datadog), team capacity, "
            "time-to-value, and 3-year TCO."
        ),
    },
    {
        "title": "Kubernetes vs serverless for a bursty ML inference workload",
        "objective": (
            "Analyze the trade-offs between EKS (Kubernetes) and AWS Lambda + SageMaker "
            "for a computer vision inference service with highly bursty traffic: "
            "0 requests at night, 50k/hour peaks during business hours. "
            "Include cold start impact, cost model, and operational complexity."
        ),
    },
    {
        "title": "Incident response automation maturity assessment",
        "objective": (
            "Assess the current state of AI-assisted incident response tooling. "
            "Which companies are using AI agents for on-call automation? "
            "What are the failure modes? What is the realistic ROI timeline? "
            "Produce a maturity model with 4 levels and criteria for each."
        ),
    },
]
