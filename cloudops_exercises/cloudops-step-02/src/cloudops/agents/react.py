from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from cloudops.core.config import settings
from cloudops.prompts.system import SYSTEM_PROMPT
from cloudops.tools.csv_tools import describe_csv, list_csv_files, query_csv
from cloudops.tools.pandas_tools import aggregate_csv, compute_correlation, timeseries_resample, top_n
from cloudops.tools.viz_tools import plot_bar, plot_histogram, plot_timeseries

TOOLS = [
    list_csv_files, describe_csv, query_csv,
    aggregate_csv, timeseries_resample, compute_correlation, top_n,
    plot_timeseries, plot_bar, plot_histogram,
]


def build_agent():
    """Return a compiled ReAct agent backed by GPT-4o and CSV-browsing tools.

    create_agent() (LangChain 1.3.1) returns a CompiledStateGraph —
    the modern LangGraph-powered replacement for the deprecated AgentExecutor.
    """
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,  # explicit so a missing key fails at startup, not mid-run
    )
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
