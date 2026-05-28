"""
Exercise 23 — Human-in-the-Loop with ReAct Agents
==================================================
Task:
    Keep the HITL checkpoint flow from exercise 12, but upgrade these nodes to
    create_agent implementations:
      - analyze
      - propose_action
      - execute_action
      - post_mortem

    Keep await_operator_approval as interrupt()-based control logic returning:
      Command(goto="execute_action" | "abort")

    Use MemorySaver checkpointer and two-phase invocation:
      - phase 1: run until interrupt
      - phase 2: resume with Command(resume=decision) and same thread_id

    Use tools:
      - run_service_diagnostic(service)
      - check_runbook_exists(incident_type)
      - get_historical_incidents(service)

Libraries:
    langgraph — StateGraph, END
    langgraph.types — interrupt, Command
    langgraph.checkpoint.memory — MemorySaver
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage
    typing — TypedDict, Optional
"""

import random
from typing import Optional, TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

INCIDENTS = [
    {
        "id": "INC-9001",
        "severity": "P0",
        "title": "API Gateway — 18% error rate, circuit breakers open",
        "details": (
            "api-gateway p99=3200ms, error_rate=18.5%, circuit breaker open for notification-service; "
            "SQS queue depth is 54k and checkout is impacted."
        ),
        "operator_decision": "approve",
    },
    {
        "id": "INC-9002",
        "severity": "P1",
        "title": "auth-service — database connection pool exhausted",
        "details": (
            "DB pool at 98/100 with rising JWT failures. Suspected connection leak after recent deploy."
        ),
        "operator_decision": "approve",
    },
    {
        "id": "INC-9003",
        "severity": "P2",
        "title": "Scheduled job — 3 consecutive failures",
        "details": (
            "nightly-report-generator failed three times with OOM "
            "(peak heap 2.1GB vs 2GB limit)."
        ),
        "operator_decision": "modify",
    },
    {
        "id": "INC-9004",
        "severity": "P1",
        "title": "payment-service — intermittent timeout spikes",
        "details": (
            "p99 latency spiked to 4.2s; 3% of transactions impacted. "
            "External processor healthy; internal DB deadlock suspected."
        ),
        "operator_decision": "approve",
    },
    {
        "id": "INC-9005",
        "severity": "P3",
        "title": "Elasticsearch indexing lag — 15-minute delay",
        "details": (
            "Log ingestion lag at 15 minutes with data nodes at 78% disk usage. "
            "No direct customer impact, but observability SLO breach."
        ),
        "operator_decision": "reject",
    },
]


class CloudOpsState(TypedDict):
    incident_id: str
    severity: str
    title: str
    details: str
    analysis: str
    proposed_action: str
    proposed_action_risk: str
    operator_decision: str
    operator_notes: str
    execution_result: str
    post_mortem: str
    status: str
