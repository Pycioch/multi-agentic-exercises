"""
Exercise 12 — Human-in-the-Loop with Checkpointing
=====================================================
Task:
    Build a cloud ops agent that analyzes a production incident, proposes a remediation
    action, then PAUSES and waits for an operator to approve, modify, or reject it.
    The graph must be durable — state survives between the two invocations.
    Pick one incident at random from INCIDENTS below.

    Nodes to build:
      - analyze          — reads the incident details, identifies root cause and blast radius
      - propose_action   — proposes ONE specific remediation action with a risk rating (LOW/MEDIUM/HIGH)
      - await_approval   — calls interrupt() to pause the graph and surface the proposal;
                           on resume, routes to execute_action or abort based on operator decision
      - execute_action   — simulates running the remediation; produces an execution log
      - post_mortem      — writes a structured post-mortem entry
      - abort            — logs the rejection and escalates

    Two-phase invocation:
      Phase 1: app.stream(initial_state, config) — runs until interrupt()
      Phase 2: app.invoke(Command(resume=operator_decision), config) — resumes from checkpoint

    Use MemorySaver as checkpointer. Link phases via thread_id in config.

Libraries:
    langgraph — StateGraph, END
    langgraph.types — interrupt, Command
    langgraph.checkpoint.memory — MemorySaver
    langchain_openai — ChatOpenAI
    langchain_core.messages — HumanMessage, SystemMessage
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

INCIDENTS = [
    {
        "id": "INC-9001",
        "severity": "P0",
        "title": "API Gateway — 18% error rate, circuit breakers open",
        "details": (
            "api-gateway: p99=3200ms, error_rate=18.5%, circuit_breaker=OPEN for notification-service. "
            "Upstream: notification-service SQS queue at 54k messages. "
            "Impact: 1800 failed requests/min. Customer-facing checkout affected."
        ),
        "operator_decision": "approve",
    },
    {
        "id": "INC-9002",
        "severity": "P1",
        "title": "auth-service — database connection pool exhausted",
        "details": (
            "auth-service: DB connection pool at 98/100. JWT validation failures increasing. "
            "5 consecutive auth failures from 10.0.0.45 (internal). "
            "Likely cause: connection leak introduced in deploy 14:30 UTC."
        ),
        "operator_decision": "approve",
    },
    {
        "id": "INC-9003",
        "severity": "P2",
        "title": "Scheduled job — 3 consecutive failures",
        "details": (
            "nightly-report-generator job failed at 02:00, 02:30, 03:00 UTC. "
            "Failure mode: OOM error (peak heap 2.1GB vs 2GB limit). "
            "Reports delayed but no user-facing impact. Previous 90 days: all successful."
        ),
        "operator_decision": "modify",
    },
    {
        "id": "INC-9004",
        "severity": "P1",
        "title": "payment-service — intermittent timeout spikes",
        "details": (
            "payment-service: p99 spiked to 4.2s at 09:45 UTC (normal: 120ms). "
            "Affects 3% of transactions. External processor (Stripe) status: all green. "
            "Suspect: internal database deadlock on orders table."
        ),
        "operator_decision": "approve",
    },
    {
        "id": "INC-9005",
        "severity": "P3",
        "title": "Elasticsearch indexing lag — 15-minute delay",
        "details": (
            "Kibana showing 15-minute lag in log ingestion. "
            "Elasticsearch cluster: 78% disk usage on data nodes. Growing 2GB/day. "
            "Impact: delayed observability only — no production service affected."
        ),
        "operator_decision": "reject",
    },
]
