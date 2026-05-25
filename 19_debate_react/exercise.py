"""
Exercise 19 — Debate with ReAct Debaters
=========================================
Task:
    Build a 3-node debate:
      advocate -> skeptic -> arbiter

    Advocate and skeptic must be create_agent workers using:
      - find_evidence(position, topic)
      - get_real_world_case(technology)

    Arbiter evaluates the exchange and returns Command:
      - goto="advocate" to continue
      - goto=END to stop with verdict

    Track all debate turns in state["messages"] with Annotated[list, operator.add].
    Stop after MAX_TURNS = 6 if no verdict appears earlier.

Libraries:
    langgraph — StateGraph, END
    langgraph.types — Command
    langchain.agents — create_agent
    langchain_core.tools — tool
    langchain_core.messages — HumanMessage, SystemMessage
    typing — TypedDict, Annotated
    operator
"""

import operator
import random
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from config import LLM, SESSION_ID, get_langfuse_handler

lf = get_langfuse_handler()

MAX_TURNS = 6

DEBATE_TOPICS = [
    "Should a 30-person startup adopt microservices architecture?",
    "Should we use a dedicated vector database over PostgreSQL + pgvector?",
    "Should regulated fintechs use open-source LLMs over hosted GPT-4-class APIs?",
    "Should AI agents have write access to production databases?",
    "Should all engineers be required to use AI coding assistants daily?",
]


class Message(TypedDict):
    role: str
    content: str


class DebateState(TypedDict):
    question: str
    advocate_position: str
    messages: Annotated[list, operator.add]
    turn_count: int
    verdict: str
    consensus_reached: bool
