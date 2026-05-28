"""
Wzorzec Reflection / ReflAct
Agent ocenia wlasne wyjscie wzgledem celu i poprawia je w petli.
Kluczowe: obowiazkowy limit iteracji — bez niego petla cykli bez konca.

Zrodlo: design_patterns/goal_state_reflection.md, design_patterns/guide_to_agentic_design_patterns.md
Odpowiada: pattern_examples/02_reflection_reflact/ z planu szkolenia
Docs: langchain_docs.md § ChatOpenAI.invoke()

ReflAct uzyskal +27.7% nad ReAct na benchmarku ALFWorld (93.3% vs 65.6%).
Reflekcja dziala, gdy model ma pokrycie domenowe — bez niego wzmacnia slepe punkty.
"""
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import re

_llm = ChatOpenAI(model="gpt-4o", temperature=0)
MAX_ITERATIONS = 3  # Obowiazkowy limit — bezpieczenstwo kosztowe


class ReflActState(TypedDict):
    task: str
    draft: str
    reflection: str
    score: int
    iteration: int


def generate(state: ReflActState) -> dict:
    feedback = f"\nUwagi do poprawy: {state['reflection']}" if state["reflection"] else ""
    msg = _llm.invoke(
        f"Zadanie: {state['task']}{feedback}\nOdpowiedz konkretnie i zwiezle."
    )
    return {"draft": msg.content, "iteration": state["iteration"] + 1}


def reflect(state: ReflActState) -> dict:
    # ReflAct: refleksja wzgledem celu, nie ogolna krytyka
    msg = _llm.invoke(
        f"Zadanie: {state['task']}\n"
        f"Aktualna odpowiedz: {state['draft']}\n\n"
        f"Oceń odpowiedz w skali 1-10 i wskaż co konkretnie wymaga poprawy. "
        f"Zacznij od liczby (np. '8/10')."
    )
    match = re.search(r"\b([1-9]|10)\b", msg.content)
    score = int(match.group(1)) if match else 5
    return {"reflection": msg.content, "score": score}


def should_continue(state: ReflActState) -> str:
    if state["score"] >= 8 or state["iteration"] >= MAX_ITERATIONS:
        return END
    return "generate"


graph = StateGraph(ReflActState)
graph.add_node("generate", generate)
graph.add_node("reflect", reflect)
graph.set_entry_point("generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue, {END: END, "generate": "generate"})

app = graph.compile()


_TASKS = [
    (
        "Przeanalizuj ponizsze objawy incydentu i napisz podsumowanie on-call: "
        "co sie stalo, jaka jest prawdopodobna przyczyna i jakie kroki naprawcze nalezy podjac.\n\n"
        "Objawy: auth-service przestaje odpowiadac co kilka minut i wymaga restartu. "
        "Przed kazdym zawieszeniem w logach pojawia sie 'HikariPool-1 - Connection is not available, "
        "request timed out after 30000ms'. Liczba aktywnych polaczen w DB rosnie stopniowo "
        "az do maksimum, po czym serwis przestaje przyjmowac ruch. Restart przywraca dzialanie "
        "na okolo 20 minut. Problem wystepuje od 14:35, nasilil sie po 16:00."
    ),
    (
        "Przeanalizuj ponizsze objawy incydentu i napisz podsumowanie on-call: "
        "co sie stalo, jaka jest prawdopodobna przyczyna i jakie kroki naprawcze nalezy podjac.\n\n"
        "Objawy: payment-service zwraca HTTP 500 dla ~30% transakcji. "
        "W logach cyklicznie pojawia sie 'ERROR: deadlock detected' na tabeli transactions. "
        "Problem nasilil sie po godzinie 18:00 gdy ruch wzrosl do 250 req/s. "
        "Przy niskim ruchu (<100 req/s) bledy nie wystepuja. "
        "Ostatni deploy byl 3 dni temu i nie dostykal warstwy bazodanowej."
    ),
    (
        "Przeanalizuj ponizsze objawy incydentu i napisz podsumowanie on-call: "
        "co sie stalo, jaka jest prawdopodobna przyczyna i jakie kroki naprawcze nalezy podjac.\n\n"
        "Objawy: api-gateway zuzywa coraz wiecej pamieci — o 14:10 bylo to 61%, "
        "o 15:40 juz 84%, o 16:55 osiagnelo 97% i zostalo zrestartowane przez orchestrator. "
        "Po restarcie pamiec wraca do 58% i znow zaczyna rosnac. "
        "Deploy v2.3.1 zostal wdrozony o 13:50. Poprzednia wersja v2.2.9 nie miala tego problemu. "
        "W v2.3.1 dodano nowy middleware do logowania naglowkow HTTP."
    ),
]


if __name__ == "__main__":
    import random
    task = random.choice(_TASKS)
    print(f"Zadanie: {task}\n")
    result = app.invoke({
        "task": task,
        "draft": "",
        "reflection": "",
        "score": 0,
        "iteration": 0,
    })
    print(f"Iteracje: {result['iteration']}, Ocena: {result['score']}/10")
    print(result["draft"])
