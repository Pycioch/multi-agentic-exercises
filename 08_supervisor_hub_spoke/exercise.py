"""
Exercise 08 — Supervisor Hub-and-Spoke
=========================================
Task:
    Build a supervisor that coordinates three specialist analysts to produce
    an investment brief on a company. Pick one subject at random from ANALYSIS_SUBJECTS.
    The supervisor reads every intermediate result before deciding who to call next.
    After all three analysts have reported, the supervisor synthesizes a final brief.

    Nodes to build:
      - supervisor         — decides which worker to call next based on what's been
                             collected so far; once all three are done, synthesizes
                             a final brief with a RECOMMENDATION (Pass/Watch/Invest)
      - fundamental_analyst — qualitative: business model, competitive moat, narrative
      - quant_analyst      — quantitative: revenue growth, valuation multiples, burn rate
      - risk_officer       — risks: top 3 risks with probability/severity, regulatory exposure,
                             headline risk rating (Low/Medium/High/Critical)

    Routing: supervisor returns Command(goto=worker | END).
    All worker edges return to supervisor (add_edge "worker" → "supervisor").
    Supervisor decides order; risk_officer gets to read the other two reports first.

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

ANALYSIS_SUBJECTS = [
    {
        "company": "Stripe",
        "sector": "FinTech / Payments Infrastructure",
        "context": (
            "Stripe is privately held, valued at ~$65B after a 2023 down round from $95B peak. "
            "Revenue ~$14B in 2023. IPO delayed. Main competitors: Adyen, Braintree, Square. "
            "Growing regulatory scrutiny in EU and UK."
        ),
    },
    {
        "company": "Snowflake",
        "sector": "Cloud Data Warehousing",
        "context": (
            "Snowflake (SNOW) ~$150 (May 2026), down from $390 ATH. FY2025 revenue $3.5B, "
            "+29% YoY but decelerating from 70%+ era. New CEO refocusing on AI data cloud. "
            "Strong competition from Databricks (private) and Google BigQuery."
        ),
    },
    {
        "company": "Klarna",
        "sector": "BNPL / Consumer Credit",
        "context": (
            "Klarna filed for IPO in 2026, targeting $15B valuation (down from $46B peak). "
            "Revenue $2.4B in 2024, profitable for first time. AI assistant handles 67% of "
            "customer service. EU Consumer Credit Directive tightening BNPL rules."
        ),
    },
    {
        "company": "Palantir",
        "sector": "AI / Government & Enterprise Data Analytics",
        "context": (
            "Palantir (PLTR) ~$25, P/E 120x on FY2025 EPS $0.21. US commercial revenue +55% YoY "
            "driven by AIP. Government contracts = 55% of revenue. CEO Peter Thiel's politics "
            "create ESG fund exclusion risk."
        ),
    },
    {
        "company": "Mistral AI",
        "sector": "Foundation Model Provider (private)",
        "context": (
            "Mistral raised $1.1B Series B at $6.2B valuation. Revenue via API + enterprise "
            "licenses + Microsoft partnership. Key product: Mistral Large 2 vs GPT-4o and Claude. "
            "European HQ — regulatory tailwind from EU AI Act. ~200 employees."
        ),
    },
]
