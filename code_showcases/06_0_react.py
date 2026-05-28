"""
Wzorzec ReAct: Thought -> Action -> Observation
Petla bazowa, na ktorej opieraja sie wszystkie pozostale wzorce.
Uzyta implementacja: langchain create_agent() — buduje petle LangGraph automatycznie.

Zrodlo: design_patterns/react_got.md, design_patterns/agentic_reasoning_frameworks.md
Odpowiada: pattern_examples/01_react/ z planu szkolenia
Docs: langchain_docs.md § create_agent()
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def search_incidents(query: str) -> str:
    """Wyszukuje incydenty Cloud-Ops po opisie lub slowach kluczowych."""
    return f"[SYMULACJA] Znaleziono 3 incydenty P1 pasujace do: '{query}'"


@tool
def get_metrics(service: str, hours: int = 24) -> str:
    """Pobiera metryki uzycia zasobow CPU i pamieci dla podanego serwisu."""
    return f"[SYMULACJA] {service}: CPU 87%, MEM 94% w ostatnich {hours}h"


# create_agent buduje graf LangGraph: model-call -> tool-call -> loop
# Thought + Action sa w jednym wywolaniu modelu; Observation to wynik narzedzia.
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[search_incidents, get_metrics],
    system_prompt=(
        "Jestes agentem Cloud-Ops. Odpowiadaj po polsku. "
        "Uzywaj narzedzi, zeby zbierac dane, zanim udzielisz odpowiedzi."
    ),
)


if __name__ == "__main__":
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            "Jakie serwisy maly wysokie zuzycie pamieci w incydentach P1 wczoraj?"
        ))]
    })
    print(result["messages"][-1].content)
