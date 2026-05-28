"""
Human-in-the-Loop z LangGraph interrupt()
Agent zawiesza wykonanie przed ryzykowna akcja i czeka na zatwierdzenie operatora.

Zrodlo: agentops/durable_execution_pattern.md
Slajd prezentacji: 3.14 — interrupt() wzorzec produkcyjny
Docs: langchain_docs.md § create_agent — interrupt_before / interrupt_after
"""
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

_llm = ChatOpenAI(model="gpt-4o", temperature=0)


class CloudOpsState(TypedDict):
    incident_id: str
    analysis: str
    proposal: str
    approved: bool
    operator: Optional[str]
    status: str
    estimated_impact: str

def analyze(state: CloudOpsState) -> dict:
    system_prompt = "Jestes asystentem CloudOps. Odpowiadaj po polsku i zwiezle."
    user_prompt = (
        "Krotko przeanalizuj incydent i zaproponuj jedno dzialanie naprawcze. "
        f"Incydent: {state['incident_id']} — wysokie zuzycie pamieci na auth-service."
    )
    msg = _llm.invoke(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )
    return {"analysis": msg.content, "proposal": msg.content}


def propose_remediation(state: CloudOpsState) -> dict:
    system_prompt = "Jestes asystentem CloudOps. Oszacuj impact wdrozenia zmiany."
    user_prompt = (
        "Podaj przewiydwany czas wdrozenia zmiany:\n"
        f"- Incydent: {state['incident_id']}\n"
        f"- Propozycja: {state['proposal']}\n"
        f"- Przewidywany czas wdrozenia zmiany: <przewidywany czas>"
    )
    estimated_impact = _llm.invoke([("system", system_prompt), ("user", user_prompt)])
    estimated_impact = estimated_impact.content.strip() or "Przewidywany czas nieznany"

    # interrupt() zawiesza graf i utrwala stan.
    # Operator widzi proposal i zatwierdza lub odrzuca.
    decision = interrupt({
        "proposal": state["proposal"],
        "incident_id": state["incident_id"],
        "estimated_impact": estimated_impact,
    })
    return {"approved": decision["approved"], "operator": decision["operator"]}


def execute_remediation(state: CloudOpsState) -> dict:
    if not state["approved"]:
        return {"status": "Wdrozenie odrzucone przez operatora"}
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return {"status": f"Wdrozenie zaakceptowane przez {state['operator']} w {timestamp}"}


graph = StateGraph(CloudOpsState)
graph.add_node("analyze", analyze)
graph.add_node("propose_remediation", propose_remediation)
graph.add_node("execute_remediation", execute_remediation)
graph.set_entry_point("analyze")
graph.add_edge("analyze", "propose_remediation")
graph.add_edge("propose_remediation", "execute_remediation")
graph.add_edge("execute_remediation", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "incident-2042"}}
    initial: CloudOpsState = {
        "incident_id": "INC-2042",
        "analysis": "",
        "proposal": "",
        "approved": False,
        "operator": None,
        "status": "pending",
    }

    # Pierwsze uruchomienie — graf zatrzymuje sie na interrupt() w propose_remediation
    print("=== Uruchamianie analizy ===")
    snapshot = app.invoke(initial, config, version="v2")

    # snapshot.interrupts zawiera wartosc przekazana do interrupt()
    interrupt_payload = snapshot.interrupts[0].value
    print(f"\nIncydent:  {interrupt_payload['incident_id']}")
    print(f"Propozycja: {interrupt_payload['proposal']}")
    print(f"Wplyw:      {interrupt_payload['estimated_impact']}")

    # Operator podejmuje decyzje — czekamy na rzeczywisty input
    raw = input("\nZatwierdz? [t/n]: ").strip().lower()
    approved = raw in ("t", "tak", "y", "yes")
    operator_name = input("Twoje imie/email: ").strip() or "operator"

    # Wznawiamy graf z decyzja operatora
    print("\n=== Wznawianie ===")
    result = app.invoke(
        Command(resume={"approved": approved, "operator": operator_name}),
        config,
        version="v2",
    )
    print(f"Status: {result.value['status']}")
