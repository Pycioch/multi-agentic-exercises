"""
Exercise 11 — Swarm / Handoff
================================
Task:
    Build a three-agent customer support swarm where agents detect when they're
    out of scope and transfer control mid-conversation using Command(goto=...).
    Pick one scenario at random from SCENARIOS below.
    Context accumulates — the receiving agent sees the full conversation history.

    Nodes to build:
      - generalist   — handles general questions; hands off to billing or technical
                       when the topic is outside its scope
      - billingagent — handles payment disputes, refunds, subscriptions;
                       hands off to technical if a technical root cause is discovered
      - technicalagent — handles API errors, webhooks, integration bugs;
                         hands off to billing if a charge issue is discovered

    Each node must:
      1. Respond to the customer
      2. Decide: RESOLVE (go to END) or TRANSFER_TO:<agent_name>
      3. Write a handoff note explaining the situation to the receiving agent
    Use Command(goto=...) to transfer, accumulate turns via Annotated[list, operator.add].
    Cap at MAX_TURNS = 8 to prevent loops.

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

SCENARIOS = [
    {
        "title": "Double charge that turns out to be an API bug",
        "opening": (
            "Hi, I was charged twice for my subscription — $49 appeared twice on my card "
            "statement on May 23rd. I want a refund for the duplicate charge."
        ),
    },
    {
        "title": "API authentication failing after plan upgrade",
        "opening": (
            "My API keys stopped working this morning. I upgraded from Pro to Enterprise "
            "yesterday and now all my authenticated requests are returning 401 errors. "
            "I haven't changed any code."
        ),
    },
    {
        "title": "Simple billing inquiry",
        "opening": (
            "Can you explain what I was charged for on my last invoice? "
            "I see a line item for 'overage fees' that I didn't expect."
        ),
    },
    {
        "title": "Webhook stopped firing after payment failure",
        "opening": (
            "Our payment webhook stopped sending events after our subscription payment failed "
            "last week. We've updated our payment method but the webhook is still silent. "
            "We're losing order fulfillment events."
        ),
    },
    {
        "title": "Account suspended — possibly payment related",
        "opening": (
            "My entire account is suspended and I can't log in. "
            "I got an email saying it was due to a payment issue but my card is valid. "
            "I need access restored immediately — our team is blocked."
        ),
    },
]
