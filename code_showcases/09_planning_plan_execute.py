"""
Wzorzec Plan-and-Execute z replanning na blad
Planer tworzy plan; executor wykonuje krok po kroku;
replanner poprawia pozostale kroki gdy poprzedni nie powiodl sie.

Roznica od ReAct: plan NAJPIERW, potem wykonanie — nie przeplatanie.

Zrodlo: design_patterns/guide_to_agentic_design_patterns.md
Odpowiada: pattern_examples/05_planning/ z planu szkolenia
Docs: langchain_docs.md § ChatOpenAI.invoke()
"""
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import re

_llm = ChatOpenAI(model="gpt-4o", temperature=0)


class Step(TypedDict):
    description: str
    status: str  # pending | done | failed
    result: Optional[str]


class PlanState(TypedDict):
    objective: str
    steps: list[Step]
    current_index: int
    final_result: str


def _parse_steps(text: str) -> list[Step]:
    return [
        {"description": re.sub(r"^\d+[\.\)]\s*", "", l.strip()), "status": "pending", "result": None}
        for l in text.strip().split("\n")
        if re.match(r"^\d+[\.\)]", l.strip())
    ]


def planner(state: PlanState) -> dict:
    msg = _llm.invoke(
        f"Rozbij cel na 3-5 konkretnych krokow. "
        f"Zwroc TYLKO ponumerowana liste (1. ... 2. ...).\nCel: {state['objective']}"
    )
    steps = _parse_steps(msg.content)
    if not steps:
        steps = [{"description": msg.content[:200], "status": "pending", "result": None}]
    return {"steps": steps, "current_index": 0}


def executor(state: PlanState) -> dict:
    idx = state["current_index"]
    step = state["steps"][idx]
    prior = "\n".join(s["result"] for s in state["steps"][:idx] if s["result"])
    steps = list(state["steps"])
    try:
        msg = _llm.invoke(
            f"Wykonaj krok: {step['description']}\nKontekst poprzednich krokow: {prior}"
        )
        steps[idx] = {**step, "status": "done", "result": msg.content}
    except Exception as e:
        steps[idx] = {**step, "status": "failed", "result": str(e)}
    return {"steps": steps, "current_index": idx + 1}


def replanner(state: PlanState) -> dict:
    done = [s for s in state["steps"] if s["status"] == "done"]
    progress = "\n".join(f"- {s['description']}: {(s['result'] or '')[:80]}" for s in done)
    msg = _llm.invoke(
        f"Popraw pozostale kroki na podstawie postepu:\n{progress}\nCel: {state['objective']}"
    )
    new_steps = _parse_steps(msg.content)
    all_steps = done + (new_steps or [{"description": "Podsumuj zebrane informacje", "status": "pending", "result": None}])
    return {"steps": all_steps, "current_index": len(done)}


def aggregator(state: PlanState) -> dict:
    context = "\n".join(
        f"Krok {i+1}: {s['result']}"
        for i, s in enumerate(state["steps"])
        if s["status"] == "done" and s["result"]
    )
    msg = _llm.invoke(f"Syntetyzuj wyniki dla celu '{state['objective']}':\n{context}")
    return {"final_result": msg.content}


def route(state: PlanState) -> str:
    idx = state["current_index"]
    if idx >= len(state["steps"]):
        return "aggregator"
    if state["steps"][idx - 1]["status"] == "failed":
        return "replanner"
    return "executor"


graph = StateGraph(PlanState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("replanner", replanner)
graph.add_node("aggregator", aggregator)
graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", route,
    {"aggregator": "aggregator", "replanner": "replanner", "executor": "executor"})
graph.add_edge("replanner", "executor")
graph.add_edge("aggregator", END)

app = graph.compile()


_OBJECTIVES = [
    (
        "Zbadaj nastepujaca sytuacje i zaproponuj plan dzialan: "
        "od wczoraj 22:15 co okolo 40 minut auth-service przestaje odpowiadac na ~3 minuty. "
        "W logach przed kazdym przestojem: 'WARN - GC overhead limit exceeded'. "
        "Heap usage tuż przed incydentem: 98%. Po restarcie wraca do 61%. "
        "Deploy byl tydzien temu. Ruch uzytkownikow nie zmienil sie."
    ),
    (
        "Zbadaj nastepujaca sytuacje i zaproponuj plan dzialan: "
        "latencja api-gateway wzrosla z 90ms do 870ms w ciagu ostatnich 6 godzin — stopniowo, nie skokowo. "
        "Upstream serwisy (auth, order, inventory) raportuja normalne czasy odpowiedzi (<100ms). "
        "CPU api-gateway: 34% (norma). MEM: 71% (norma). "
        "Liczba aktywnych polaczen do upstream: wzrosla z 40 do 380. Limit: 400. "
        "Timeout na polaczeniach zaczal sie pojawiac 2 godziny temu."
    ),
    (
        "Zbadaj nastepujaca sytuacje i zaproponuj plan dzialan: "
        "payment-service i auth-service sa niedostepne jednoczesnie od 11:03. "
        "payment-service log: 'Unable to acquire JDBC Connection'. "
        "auth-service log: 'Connection refused' do serwisu token-store (Redis). "
        "Wspolna infrastruktura: obydwa korzystaja z tego samego klastra PostgreSQL i Redis. "
        "PostgreSQL: 412/400 aktywnych polaczen (przekroczony limit). Redis: dziala, ping OK."
    ),
]


if __name__ == "__main__":
    import random
    objective = random.choice(_OBJECTIVES)
    print(f"Cel: {objective}\n")
    result = app.invoke({
        "objective": objective,
        "steps": [],
        "current_index": 0,
        "final_result": "",
    })
    print(result["final_result"])
