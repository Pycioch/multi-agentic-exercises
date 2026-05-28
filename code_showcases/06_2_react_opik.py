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
from opik.integrations.langchain import OpikTracer


_INCIDENT_SETS = [
    ["INC-1023: auth-service — wyczerpana pula polaczen DB",
     "INC-1024: payment-service — timeout przy walidacji kart",
     "INC-1025: api-gateway — spike latencji do 8s"],
    ["INC-2041: order-service — OOM crash na nodzie prod-3",
     "INC-2042: inventory-service — deadlock na tabeli stock_reservations",
     "INC-2043: notification-service — kolejka SQS przepelniona (>50k msg)"],
    ["INC-3087: auth-service — MEM 94%, restart co 20min",
     "INC-3088: payment-service — CPU 87%, degradacja throughput o 60%",
     "INC-3089: reporting-service — zapytania SQL >30s, blokada I/O"],
]


@tool
def search_incidents(query: str) -> str:
    """Wyszukuje incydenty Cloud-Ops po opisie lub slow kluczowych."""
    import random
    incidents = random.choice(_INCIDENT_SETS)
    names = "\n".join(f"  - {i}" for i in incidents)
    return f"[SYMULACJA] Znaleziono 3 incydenty P1 pasujace do: '{query}':\n{names}"


@tool
def get_metrics(service: str, hours: int = 24) -> str:
    """Pobiera metryki uzycia zasobow CPU i pamieci dla podanego serwisu."""
    import random
    cpu = random.randint(55, 99)
    mem = random.randint(55, 99)
    return f"[SYMULACJA] {service}: CPU {cpu}%, MEM {mem}% w ostatnich {hours}h"


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
    tracer = OpikTracer(project_name="multi_agent_training")
    result = agent.invoke(
        {"messages": [HumanMessage(content=(
            "Jakie serwisy maly wysokie zuzycie pamieci w incydentach P1 wczoraj?"
        ))]},
        config={"callbacks": [tracer]},
    )
    print(result["messages"][-1].content)
    tracer.flush()
