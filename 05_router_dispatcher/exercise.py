"""
Exercise 05 — Router / Dispatcher
====================================
Task:
    Build a router that classifies an incoming support message into one department
    and routes it to the matching specialist agent. Pick one message at random.
    The router makes ONE decision and steps aside — no aggregation, no follow-up.

    Nodes to build:
      - router     — classifies the message into: billing / technical / account / sales;
                     returns Command(goto=department) with the routing decision
      - billing    — responds to payment disputes, refunds, subscription charges
      - technical  — responds to API errors, webhook failures, integration bugs
      - account    — responds to password reset, access management, team seats
      - sales      — responds to pricing, upgrade queries, feature availability

    Each specialist has its own knowledge base (SPECIALIST_KB below).
    After the specialist responds, the graph ends.

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
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

SUPPORT_MESSAGES = [
    "I was charged twice for my subscription this month. I need a refund ASAP.",
    "My API calls are returning 429 errors even though I'm well below my rate limit. "
    "I've checked the docs and I'm sending the correct headers.",
    "I forgot my password and the reset email isn't arriving. Checked spam folder.",
    "We're a team of 15 and I want to understand if the Enterprise plan makes sense "
    "compared to paying per seat on Pro. What's the break-even point?",
    "The webhook I set up for order.completed events stopped firing after your "
    "maintenance window yesterday. The endpoint is healthy on our side.",
    "Can I add a second admin to my account? I need my CTO to have full access "
    "without sharing my login.",
    "You charged my old card that was cancelled last week. I've added a new card "
    "but the invoice still shows the cancelled one as the payment method.",
    "I'm evaluating your product vs a competitor. Does your API support "
    "GraphQL queries or only REST? Also do you have a sandbox environment?",
]

SPECIALIST_KB = {
    "billing": (
        "- Refund policy: full refund within 30 days of charge, no questions asked.\n"
        "- Double charges: always genuine bug — initiate refund immediately + ticket to finance.\n"
        "- Payment method updates: go to Settings → Billing → Payment Methods.\n"
        "- Invoice disputes: SLA 3 business days for resolution.\n"
        "- Contact: billing@company.com or in-app chat."
    ),
    "technical": (
        "- 429 rate limit errors: check X-RateLimit-Remaining header; burst window is 60s.\n"
        "  Common cause: parallel requests sharing one API key. Solution: exponential backoff.\n"
        "- Webhook failures: check webhook_logs table in dashboard; test with /webhooks/test.\n"
        "  Post-maintenance: webhooks auto-resume — if not, re-save the endpoint URL.\n"
        "- API docs: https://docs.company.com/api/v2\n"
        "- Status page: https://status.company.com"
    ),
    "account": (
        "- Password reset: link valid for 15 minutes; whitelist noreply@company.com.\n"
        "  Alternative: request magic link from login page.\n"
        "- Multi-admin: Settings → Team → Invite Member → select 'Admin' role.\n"
        "- SSO: available on Enterprise plan with SAML 2.0 or OIDC."
    ),
    "sales": (
        "- Pro plan: $49/seat/month, up to 10 seats.\n"
        "- Enterprise plan: $399/month flat for unlimited seats + SSO + dedicated CSM.\n"
        "  Break-even vs Pro: at 9+ seats, Enterprise is cheaper.\n"
        "- Sandbox environment: available on all paid plans at sandbox.company.com.\n"
        "- GraphQL: not currently supported; REST only with OpenAPI 3.0 spec available.\n"
        "- Free trial: 14 days on Pro plan, no credit card required."
    ),
}
