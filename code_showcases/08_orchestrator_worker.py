"""
Wzorzec Orchestrator-Worker z rownoleglym fan-out (Send API)
Orkiestrator dzieli zadanie na podzadania; kazdy worker dziala niezaleznie.
Worker zwraca STRING podsumowania — nie pelny transkrypt.

Zrodlo: design_patterns/anthropic_multi_agent_blueprint.md, design_patterns/multi_agent_patterns.md
Odpowiada: pattern_examples/04_orchestrator_worker/ z planu szkolenia
Docs: langchain_docs.md § ChatOpenAI, create_agent

+81% na zadaniach rownoleglizowalnych (Finance-Agent benchmark).
-70% na zadaniach sekwencyjnych — kazde przekazanie to stratna kompresja.
Badanie Anthropic: izolowane subagenty daly +90.2% w systemie badawczym.
"""
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model="gpt-4o", temperature=0)


class OrchestratorState(TypedDict):
    request: str
    subtasks: list[str]
    results: Annotated[list[str], operator.add]


class WorkerState(TypedDict):
    subtask: str


def orchestrator(state: OrchestratorState) -> dict:
    msg = _llm.invoke(
        f"Podziel zadanie na 2-4 niezalezne podzadania (jedno na linii, bez numeracji):\n"
        f"{state['request']}"
    )
    subtasks = [line.strip() for line in msg.content.strip().split("\n") if line.strip()]
    return {"subtasks": subtasks}


def dispatch(state: OrchestratorState) -> list:
    # Send() uruchamia kazdy worker rownolegle
    return [Send("worker", {"subtask": task}) for task in state["subtasks"]]


def worker(state: WorkerState) -> dict:
    msg = _llm.invoke(
        f"Wykonaj to podzadanie i zwroc TYLKO krotkie podsumowanie wynikow "
        f"(nie pelny transkrypt):\n{state['subtask']}"
    )
    # Kluczowa zasada: worker zwraca string, nie cala history rozmowy
    return {"results": [msg.content]}


def aggregator(state: OrchestratorState) -> dict:
    combined = "\n---\n".join(state["results"])
    msg = _llm.invoke(
        f"Syntetyzuj wyniki podzadan w spojna odpowiedz:\n{combined}"
    )
    return {"results": [msg.content]}


graph = StateGraph(OrchestratorState)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("aggregator", aggregator)
graph.set_entry_point("orchestrator")
graph.add_conditional_edges("orchestrator", dispatch, then="aggregator")
graph.add_edge("aggregator", END)

app = graph.compile()


_REQUESTS = [
    (
        "O 14:22 wdrozono nowa wersje order-service. O 14:35 zaczely napływac alerty. "
        "Objawy: CPU api-gateway wzrosl z 28% do 79%, latencja p99 wzrosla z 140ms do 1200ms, "
        "error rate order-service wzrosl z 0.1% do 8.3%, "
        "baza danych order-service: active connections 387/400. "
        "Zbadaj niezaleznie metryki CPU, latencji i bazy danych — "
        "ustal czy deploy jest przyczyna i co dokladnie sie psuje."
    ),
    (
        "Od 03:41 w nocy trzy serwisy zgłaszaja anomalie: "
        "auth-service: blad 'read timeout' do Redis co ~90 sekund, "
        "payment-service: sporadyczne HTTP 503, srednio 12 bledow na minute, "
        "notification-service: kolejka wzrosla z 200 do 41000 wiadomosci w ciagu 2 godzin. "
        "Zadnego deployu nie bylo od 48 godzin. "
        "Zbadaj kazdy serwis niezaleznie i wskaż czy to wspolna przyczyna czy niezalezne problemy."
    ),
    (
        "Raport wydajnosciowy po migracji bazy danych PostgreSQL 14 -> 16: "
        "zapytania SELECT na tabeli orders: przed migracją srednia 18ms, po migracji 94ms, "
        "zapytania INSERT: przed 12ms, po 11ms (bez zmian), "
        "EXPLAIN ANALYZE pokazuje 'Seq Scan' tam gdzie wczesniej byl 'Index Scan', "
        "rozmiar bazy: 340GB, autovacuum ostatnio uruchomiony 6 dni temu. "
        "Zbadaj niezaleznie: plany zapytan, stan indeksow, statystyki tabel i konfiguracje PostgreSQL 16 — "
        "zidentyfikuj przyczyne regresji i zaproponuj naprawe."
    ),
]


if __name__ == "__main__":
    import random
    request = random.choice(_REQUESTS)
    print(f"Zadanie: {request}\n")
    result = app.invoke({
        "request": request,
        "subtasks": [],
        "results": [],
    })
    print(result["results"][-1])
