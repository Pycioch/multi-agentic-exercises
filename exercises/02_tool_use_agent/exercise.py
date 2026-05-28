"""
Exercise 02 — Tool Use / ReAct Agent
=======================================
Task:
    Build a ReAct agent that diagnoses production incidents by calling tools.
    Pick one incident at random from INCIDENTS below and let the agent investigate.
    The agent should chain tool calls as needed and finish with a concrete recommendation.

    Tools to build:
      - search_runbook(issue: str)         — looks up RUNBOOKS by keyword
      - check_service_health(service: str) — returns metrics from SERVICE_HEALTH
      - read_recent_logs(service: str, n)  — returns lines from LOG_SAMPLES

Libraries:
    langchain.agents — create_agent
    langchain_openai — ChatOpenAI
    langchain_core.tools — tool  (decorator)
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

RUNBOOKS = {
    "database connection pool exhausted": (
        "Runbook DB-04: Check PG_POOL_SIZE env var. "
        "Restart connection pool with: systemctl restart pg-bouncer. "
        "Alert DBA if pool exhaustion happens > 3 times/hour."
    ),
    "memory leak": (
        "Runbook MEM-11: Capture heap dump with 'jmap -dump'. "
        "Restart JVM if heap > 90%. Investigate leaking class in next sprint."
    ),
    "high latency": (
        "Runbook LAT-07: Check downstream service SLAs. "
        "Enable request tracing. Profile slow endpoints with py-spy."
    ),
    "disk full": (
        "Runbook DISK-02: Run 'df -h'. Clear /tmp and old logs. "
        "Extend volume if < 10% free. Alert infra team."
    ),
    "authentication failure": (
        "Runbook AUTH-09: Check JWT expiry settings. "
        "Rotate secrets if compromised. Verify LDAP connectivity."
    ),
    "cpu spike": (
        "Runbook CPU-03: Run 'top -c' and 'perf top'. "
        "Check for runaway processes. Scale horizontally if > 80% sustained."
    ),
}

SERVICE_HEALTH = {
    "auth-service":         {"status": "degraded",  "latency_ms": 842,  "error_rate_pct": 7.3,  "uptime_pct": 99.1},
    "payment-service":      {"status": "healthy",   "latency_ms": 120,  "error_rate_pct": 0.1,  "uptime_pct": 99.99},
    "api-gateway":          {"status": "critical",  "latency_ms": 3200, "error_rate_pct": 18.5, "uptime_pct": 97.2},
    "order-service":        {"status": "healthy",   "latency_ms": 95,   "error_rate_pct": 0.3,  "uptime_pct": 99.95},
    "inventory-service":    {"status": "degraded",  "latency_ms": 510,  "error_rate_pct": 4.1,  "uptime_pct": 99.5},
    "notification-service": {"status": "critical",  "latency_ms": 8900, "error_rate_pct": 42.0, "uptime_pct": 91.0},
}

LOG_SAMPLES = {
    "auth-service": [
        "ERROR 2026-05-25T09:12:04Z [auth-service] JWT validation failed: token expired",
        "WARN  2026-05-25T09:12:06Z [auth-service] LDAP connection timeout after 800ms",
        "ERROR 2026-05-25T09:12:09Z [auth-service] 5 consecutive auth failures from 10.0.0.45",
        "ERROR 2026-05-25T09:12:11Z [auth-service] Database pool: 98/100 connections used",
    ],
    "api-gateway": [
        "ERROR 2026-05-25T09:15:01Z [api-gateway] Upstream timeout: payment-service (3100ms)",
        "ERROR 2026-05-25T09:15:03Z [api-gateway] Circuit breaker OPEN for notification-service",
        "WARN  2026-05-25T09:15:05Z [api-gateway] Rate limit exceeded for tenant-7731",
        "ERROR 2026-05-25T09:15:07Z [api-gateway] 503 returned to 1842 requests in last 60s",
    ],
    "notification-service": [
        "ERROR 2026-05-25T09:18:00Z [notif] SQS queue depth: 54892 messages (threshold: 1000)",
        "ERROR 2026-05-25T09:18:04Z [notif] Worker thread pool exhausted (64/64 busy)",
        "WARN  2026-05-25T09:18:08Z [notif] Email delivery failure rate: 38%",
        "ERROR 2026-05-25T09:18:12Z [notif] OutOfMemoryError in thread pool-worker-12",
    ],
}

INCIDENTS = [
    "INC-3041: Users reporting login failures since 09:10. Auth team says JWTs are invalid.",
    "INC-3042: API Gateway returning 503 to 18% of requests. Spike started 15 minutes ago.",
    "INC-3043: Notification service queue growing uncontrollably. 50k+ messages pending.",
    "INC-3044: Payment service team reports everything is green on their end, "
               "but customers say checkout is slow. Investigate the full request path.",
    "INC-3045: On-call engineer asks: which of our services is currently in the worst shape? "
               "Give me a prioritized list with evidence.",
    "INC-3046: Post-mortem prep — summarize what happened with the auth service this morning. "
               "Include logs and health data.",
]
