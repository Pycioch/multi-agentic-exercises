"""
Exercise 14 — Sequential Pipeline with ReAct Stages
====================================================
Task:
    Rebuild the 4-stage linear pipeline from exercise 01, but each stage is
    a create_agent worker with tools:
      - extractor
      - writer
      - fact_checker
      - publisher

    Keep graph topology strictly sequential:
      extractor -> writer -> fact_checker -> publisher -> END

    Use search_archive and verify_claim as shared tools.
    Pick one snippet at random from RAW_RESEARCH_SNIPPETS below.

Libraries:
    langgraph — StateGraph, END
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage
"""

import random
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

RAW_RESEARCH_SNIPPETS = [
    """\
    Klarna's AI assistant handled 2.3 million customer conversations in its first month,
    doing the work of 700 full-time agents. Average resolution time dropped from
    11 minutes to 2 minutes. Customer satisfaction held steady at pre-AI levels.
    The project cost roughly $3M to deploy and saved an estimated $40M annually.
    """,
    """\
    A 2025 study by Stanford HAI found that GPT-4-class models hallucinate on
    medical questions roughly 14% of the time even when retrieval-augmented.
    Fine-tuned models on PubMed cut that to 6%. Smaller 7B models hallucinated at 31%.
    """,
    """\
    LangGraph surpassed CrewAI in GitHub stars in early 2026, reaching 95k stars.
    Enterprise adoption doubled year-over-year across major tech companies.
    """,
    """\
    OpenAI's o3 model scored 87.5% on the ARC-AGI benchmark, compared to 4%
    for GPT-4o. The model reportedly averages around $15 per problem solved.
    """,
]


class PipelineState(TypedDict):
    raw_input: str
    key_facts: str
    polished_paragraph: str
    fact_check_notes: str
    final_summary: str
