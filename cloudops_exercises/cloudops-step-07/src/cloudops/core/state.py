from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state passed through every node in the graph.

    Introduced here (step 3) even before LangGraph graphs appear — step 5+
    will import and extend this definition rather than redefine it.
    """

    messages: Annotated[list[BaseMessage], add_messages]
