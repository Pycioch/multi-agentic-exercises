from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from cloudops.core.config import settings
from cloudops.prompts.system import SYSTEM_PROMPT
from cloudops.tools.csv_tools import describe_csv, list_csv_files, query_csv
from cloudops.tools.pandas_tools import aggregate_csv, compute_correlation, timeseries_resample, top_n
from cloudops.tools.viz_tools import plot_bar, plot_histogram, plot_timeseries


@tool
def ask_human(question: str) -> str:
    """Pause and ask the human operator a clarification question before continuing.

    Use this ONLY when genuinely blocked — i.e., the question is ambiguous AND
    the answer would be materially different depending on the missing information.

    Do NOT use it when:
    - The question already specifies what is needed (datacenter, host, date, ID)
    - You can reasonably infer the scope from context
    - The question asks for aggregate data across all datacenters/services

    Use it when:
    - The question says "the service" or "the incident" without identifying which
    - Multiple datacenters are possible and the user seems to want just one

    Args:
        question: A short, specific question (e.g. "Which datacenter? DC-A, DC-B, or DC-C?").
    """
    # interrupt() suspends the graph and surfaces the payload to the CLI.
    # The CLI collects the human's answer and resumes via Command(resume=answer).
    return interrupt({"question": question})


TOOLS = [
    list_csv_files, describe_csv, query_csv,
    aggregate_csv, timeseries_resample, compute_correlation, top_n,
    plot_timeseries, plot_bar, plot_histogram,
    ask_human,
]


def build_agent():
    """Return a compiled ReAct agent with HITL support via MemorySaver checkpointer."""
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,  # explicit so a missing key fails at startup, not mid-run
    )
    # MemorySaver keeps state in-process — sufficient for HITL in a single CLI session.
    # Step 7 replaces this with AsyncPostgresSaver for durable cross-process persistence.
    checkpointer = MemorySaver()
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
