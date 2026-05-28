"""
Exercise 04 — Evaluator-Optimizer
====================================
Task:
    Build a loop where a generator writes a product description and an independent
    evaluator scores it on 4 rubric dimensions (25 pts each, 100 total).
    Loop until score ≥ 82/100 or 3 iterations pass. Pick one brief at random.

    Nodes to build:
      - generate — writes a 3-paragraph product description (150-200 words);
                   on reruns, receives structured feedback from the previous evaluation
      - evaluate — scores the description on Clarity / Specificity / SEO / Call-to-Action;
                   produces dimension-level scores and improvement notes
                   (this is an INDEPENDENT judge — not the same model self-critiquing)

    Routing: use Command(goto="generate" | END) inside evaluate.
    Parse the total score from the evaluator output ("Total: XX/100").

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

PRODUCT_BRIEFS = [
    {
        "product": "DataSync Pro",
        "category": "B2B SaaS / data integration",
        "audience": "CTOs and data engineers at mid-market companies",
        "key_features": [
            "Real-time sync between 200+ data sources",
            "No-code pipeline builder with drag-and-drop UI",
            "99.99% uptime SLA with automatic failover",
            "SOC 2 Type II certified",
        ],
        "price_point": "$499/month per workspace",
        "primary_benefit": "Eliminate manual data pipelines and ETL bottlenecks",
    },
    {
        "product": "FocusBrew Coffee Subscription",
        "category": "D2C / food & beverage",
        "audience": "Remote workers and freelancers aged 25–40",
        "key_features": [
            "Specialty single-origin beans, roasted to order",
            "Nootropic blend with L-theanine for smooth energy",
            "Personalized grind size and roast level",
            "Skip, pause, or cancel anytime",
        ],
        "price_point": "$29/month for 12 oz bag",
        "primary_benefit": "Sustained focus without the afternoon crash",
    },
    {
        "product": "LegalDraft AI",
        "category": "Legal tech / AI assistant",
        "audience": "Solo practitioners and small law firms",
        "key_features": [
            "Drafts NDAs, contracts, and demand letters in minutes",
            "Jurisdiction-aware clause library (US, UK, EU)",
            "Redline comparison against uploaded templates",
            "Integrates with Clio and MyCase",
        ],
        "price_point": "$149/month per attorney",
        "primary_benefit": "Cut contract drafting time by 70% without sacrificing accuracy",
    },
    {
        "product": "TrailGuard GPS Watch",
        "category": "Consumer hardware / outdoor fitness",
        "audience": "Serious hikers and ultra-marathon runners",
        "key_features": [
            "Topographic maps offline, 50k trails pre-loaded",
            "30-day battery life in GPS mode",
            "Emergency SOS via satellite (no cell required)",
            "Health metrics: VO2 max, altitude acclimatization score",
        ],
        "price_point": "$449 one-time",
        "primary_benefit": "Never get lost, even off the grid",
    },
    {
        "product": "ResumeCoach AI",
        "category": "EdTech / career tools",
        "audience": "Mid-career professionals changing industries",
        "key_features": [
            "ATS score checker against real job postings",
            "Skill gap analysis with learning path recommendations",
            "1-click tailoring to specific job descriptions",
            "Mock interview simulator with AI feedback",
        ],
        "price_point": "$19/month or $149/year",
        "primary_benefit": "Land more interviews with a resume that actually gets read",
    },
    {
        "product": "ColdRoom Smart Freezer",
        "category": "Smart home / kitchen appliance",
        "audience": "Home chefs and meal-prep enthusiasts",
        "key_features": [
            "AI-powered food inventory tracking via internal camera",
            "Expiry alerts and recipe suggestions based on contents",
            "Zone temperature control (fresh, chill, freeze)",
            "Energy usage 40% below EU A++ standard",
        ],
        "price_point": "$1,299 one-time",
        "primary_benefit": "Stop wasting food and always know what's in your freezer",
    },
]
