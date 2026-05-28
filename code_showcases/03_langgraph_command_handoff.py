"""
Handoff miedzy agentami w LangGraph — Command(goto=...) vs conditional_edges

=== Command(goto=...) ===

Command to obiekt zwracany bezposrednio z wezla grafu. Laczy w sobie dwie operacje
atomicznie: aktualizacje stanu (update=) i decyzje o nastepnym wezle (goto=).

Wezel sam decyduje, dokad przekazac kontrole — logika routingu jest WEWNATRZ wezla.
Wezel nie tylko wykonuje prace, ale rowniez wie, kogo wezwac nastepnie. Zachowuje
sie jak prawdziwy orkiestrator: klasyfikuje problem i natychmiast deleguje.

Dlaczego atomowosc ma znaczenie:
  Bez Command sekwencja wyglada tak:
    1. wezel zwraca dict  →  stan zaktualizowany
    2. [stan posredni — nowe wartosci, brak decyzji co dalej]
    3. funkcja routujaca czyta stan  →  wybiera nastepnik
  Miedzy krokiem 1 a 3 istnieje chwila, gdy stan ma juz nowe wartosci,
  ale nikt jeszcze nie zdecydowal, co z nimi zrobic. W prostych grafach
  to nie problem, ale gdy wezly dzialaja rownolegle lub routing zalezy
  od wielu pol jednoczesnie, ta niespojnosc moze prowadzic do bledow.
  Command eliminuje ten stan posredni — update i goto sa jedna operacja.

Kiedy uzywac Command:
- Gdy wezel jest decydentem — sam zna kontekst potrzebny do wyboru nastepnika.
- Gdy aktualizacja stanu i wybor nastepnika musza byc spojne (np. zapisujesz
  reasoning i na jego podstawie wybierasz agenta — nie chcesz, zeby routing
  odbywal sie na starym stanie).
- Gdy routing zalezy od wyniku obliczen WEWNATRZ wezla (np. odpowiedzi LLM),
  a nie od prostej wartosci pola stanu.
- Typowe zastosowanie: orkiestrator, ktory klasyfikuje i deleguje jednoczesnie.

=== conditional_edges ===

add_conditional_edges() definiuje routing POZA wezlem, w osobnej funkcji routujacej.
Wezel aktualizuje stan zwracajac dict, a zewnetrzna funkcja routujaca czyta stan
i zwraca nazwe nastepnika. Sa to dwie odrebne operacje — rozdzielenie odpowiedzialnosci.

Kiedy uzywac conditional_edges:
- Gdy logika routingu jest prosta i zalezna tylko od wartosci pola stanu
  (np. state["complexity"] == "simple").
- Gdy chcesz wyraznie rozdzielic "co wezel robi" od "gdzie idzie kontrola" —
  latwe do testowania i modyfikowania niezaleznie.
- Gdy ta sama logika routingu obsluguje wiele wezlow w grafie.
- Gdy routing to konfiguracja grafu, a nie logika biznesowa agenta.

Porownanie na przykladzie z tego pliku:

  # Command — klasyfikacja i delegacja to jedna operacja
  def orchestrator_command(state):
      classification = _classifier.invoke(...)      # LLM decyduje
      return Command(
          goto="simple_agent" if ... else "complex_agent",   # routing tu
          update={"complexity": ..., "reasoning": ...},
      )

  # conditional_edges — klasyfikacja i routing to dwa odrebne kroki
  def orchestrator_edges(state):
      classification = _classifier.invoke(...)      # LLM decyduje
      return {"complexity": ..., "reasoning": ...}  # tylko update

  def route_by_complexity(state):                   # routing osobno
      return "simple_agent" if state["complexity"] == "simple" else "complex_agent"

=== Przekazywanie kontekstu miedzy agentami ===

Stan (CloudOpsState) jest wspoldzielonym buforem — kazdy wezel moze go czytac
i wzbogacac. Nie jest to tylko "pamiec", ale kanal komunikacji miedzy agentami.

W tym przykladzie:
- Orkiestrator zapisuje pole "reasoning" (uzasadnienie klasyfikacji) do stanu.
- Subagenci (simple_agent, complex_agent) czytaja to pole i wlaczaja je do promptu,
  dzieki czemu wiedza, DLACZEGO zostaly wywolane i moga dostosowac styl odpowiedzi.
- Finalny stan zwrocony do klienta zawiera pelny slad decyzji: query, complexity,
  reasoning i result — mozna go uzyc do debugowania, logowania lub oceny.

Zrodlo: agent_communication/handoff.md
Slajd prezentacji: 7.14 — Command(goto=...)
"""
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Command

_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# Schemat Pydantic — structured output orkiestratora
# ---------------------------------------------------------------------------

class QueryClassification(BaseModel):
    complexity: Literal["simple", "complex"] = Field(
        description="'simple' dla pojedynczych pytan, 'complex' dla pytan wymagajacych korelacji lub wielu zrodel"
    )
    reasoning: str = Field(description="Krotkie uzasadnienie decyzji w 1-2 zdaniach")


_classifier = _llm.with_structured_output(QueryClassification)


# ---------------------------------------------------------------------------
# Stan wspoldzielony — przekazywany miedzy wszystkimi agentami
# ---------------------------------------------------------------------------

class CloudOpsState(TypedDict):
    query: str
    complexity: Literal["simple", "complex"]
    reasoning: str   # orkiestrator zapisuje uzasadnienie; subagenci je czytaja
    result: str


# ===========================================================================
# WZORZEC 1: Command(goto=...) — routing WEWNATRZ wezla
# ===========================================================================

def orchestrator_command(state: CloudOpsState) -> Command:
    """Klasyfikuje zapytanie i atomicznie aktualizuje stan oraz wybiera nastepnik."""
    classification: QueryClassification = _classifier.invoke(
        f"Sklasyfikuj zlozonosc zapytania Cloud-Ops:\n\n{state['query']}"
    )
    return Command(
        goto="simple_agent" if classification.complexity == "simple" else "complex_agent",
        update={
            "complexity": classification.complexity,
            "reasoning": classification.reasoning,
        },
    )


def simple_agent(state: CloudOpsState) -> dict:
    """Subagent dla prostych zapytan. Czyta reasoning orkiestratora z kontekstu."""
    msg = _llm.invoke(
        f"Kontekst od orkiestratora: {state['reasoning']}\n\n"
        f"Odpowiedz zwiezle na pytanie Cloud-Ops: {state['query']}"
    )
    return {"result": msg.content}


def complex_agent(state: CloudOpsState) -> dict:
    """Subagent dla zlozonych zapytan. Czyta reasoning orkiestratora z kontekstu."""
    msg = _llm.invoke(
        f"Kontekst od orkiestratora: {state['reasoning']}\n\n"
        f"To zapytanie wymaga analizy wielu zrodel. Przeanalizuj doglebnie: {state['query']}"
    )
    return {"result": msg.content}


graph_command = StateGraph(CloudOpsState)
graph_command.add_node("orchestrator", orchestrator_command)
graph_command.add_node("simple_agent", simple_agent)
graph_command.add_node("complex_agent", complex_agent)
graph_command.add_edge("simple_agent", END)
graph_command.add_edge("complex_agent", END)
graph_command.set_entry_point("orchestrator")

app_command = graph_command.compile()


# ===========================================================================
# WZORZEC 2: conditional_edges — routing POZA wezlem
# ===========================================================================

def orchestrator_edges(state: CloudOpsState) -> dict:
    """Klasyfikuje zapytanie i zapisuje wynik do stanu. Routing decyduje osobna funkcja."""
    classification: QueryClassification = _classifier.invoke(
        f"Sklasyfikuj zlozonosc zapytania Cloud-Ops:\n\n{state['query']}"
    )
    return {
        "complexity": classification.complexity,
        "reasoning": classification.reasoning,
    }


def route_by_complexity(state: CloudOpsState) -> Literal["simple_agent", "complex_agent"]:
    """Funkcja routujaca — czyta stan i zwraca nazwe nastepnego wezla."""
    return "simple_agent" if state["complexity"] == "simple" else "complex_agent"


graph_edges = StateGraph(CloudOpsState)
graph_edges.add_node("orchestrator", orchestrator_edges)
graph_edges.add_node("simple_agent", simple_agent)
graph_edges.add_node("complex_agent", complex_agent)
graph_edges.add_conditional_edges("orchestrator", route_by_complexity)
graph_edges.add_edge("simple_agent", END)
graph_edges.add_edge("complex_agent", END)
graph_edges.set_entry_point("orchestrator")

app_edges = graph_edges.compile()


# ===========================================================================
# Uruchomienie obu wzorow
# ===========================================================================

QUERY_SIMPLE = "Ile incydentow P0 bylo w DC-A w Q1 2025?"
QUERY_COMPLEX = (
    "Czy incydenty z 15 marca koreluja z deployem serwisu payment-service? "
    "Porownaj trzy strumienie metryk: CPU, pamiec i latencje dla wszystkich dotkietych serwisow."
)

# Wybierz zapytanie do testu:
QUERY = QUERY_SIMPLE
# QUERY = QUERY_COMPLEX


if __name__ == "__main__":
    initial = {"query": QUERY, "complexity": "simple", "reasoning": "", "result": ""}

    print("=== WZORZEC 1: Command(goto=...) ===\n")
    result = app_command.invoke(initial)
    print(f"[{result['complexity'].upper()}] Reasoning: {result['reasoning']}")
    print(f"Odpowiedz: {result['result']}\n")

    print("=== WZORZEC 2: conditional_edges ===\n")
    result = app_edges.invoke(initial)
    print(f"[{result['complexity'].upper()}] Reasoning: {result['reasoning']}")
    print(f"Odpowiedz: {result['result']}\n")
