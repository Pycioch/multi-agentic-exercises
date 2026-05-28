"""
Wzorzec Tree of Thoughts (ToT) — przeszukiwanie BFS
Model generuje wiele galeziowych mysli, ocenia je i eksploruje najlepsze.
Uczciwosc kosztowa: ToT zuzywa wielokrotnie wiecej tokenow niz ReAct.

Zrodlo: design_patterns/react_got.md, design_patterns/agentic_reasoning_frameworks.md
Odpowiada: pattern_examples/06_tree_of_thoughts/ z planu szkolenia
Docs: langchain_docs.md § ChatOpenAI.invoke()

Kiedy uzywac: problemy wymagajace eksploracji przestrzeni rozwiazan (planowanie,
dowody matematyczne). Kiedy NIE uzywac: wiekszosc codziennych zadan — ReAct wystarczy.
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model="gpt-4o", temperature=0)
BEAM_WIDTH = 2   # ile galezi rozwijamy na kazdym poziomie
MAX_DEPTH = 3    # maksymalna glebokosc drzewa


def generate_thoughts(problem: str, context: str, n: int) -> list[str]:
    """Generuje n alternatywnych nastepnych krokow mysli."""
    msg = _llm.invoke(
        f"Problem: {problem}\n"
        f"Dotychczasowy tok rozumowania: {context or '(brak)'}\n\n"
        f"Wygeneruj dokladnie {n} rozne nastepne kroki rozumowania (numerowane 1..{n})."
    )
    lines = [l.strip() for l in msg.content.split("\n") if l.strip()]
    thoughts = [l for l in lines if l[:2] in [f"{i}." for i in range(1, n + 1)]]
    return thoughts[:n] if len(thoughts) >= n else lines[:n]


def evaluate_thought(problem: str, thought_chain: str) -> float:
    """Ocenia obiecujacosć ścieżki mysli w skali 0-1."""
    msg = _llm.invoke(
        f"Problem: {problem}\n"
        f"Tok rozumowania: {thought_chain}\n\n"
        f"Oceń czy ten tok prowadzi do rozwiazania. Odpowiedz TYLKO liczba 0.0-1.0."
    )
    try:
        return float(msg.content.strip().split()[0])
    except (ValueError, IndexError):
        return 0.5


def is_solution(problem: str, thought_chain: str) -> tuple[bool, str]:
    """Sprawdza czy tok rozumowania doszedl do rozwiazania."""
    msg = _llm.invoke(
        f"Problem: {problem}\n"
        f"Tok rozumowania: {thought_chain}\n\n"
        f"Czy to kompletne rozwiazanie? Zacznij od TAK lub NIE, potem wyjasnij."
    )
    is_done = msg.content.strip().upper().startswith("TAK")
    return is_done, msg.content


def tree_of_thoughts_bfs(problem: str) -> str:
    """BFS przez drzewo mysli z beam search."""
    # Kazdy element to ciag mysli prowadzacy do danego wezla
    beam: list[str] = [""]

    for depth in range(MAX_DEPTH):
        candidates: list[tuple[float, str]] = []

        for thought_chain in beam:
            new_thoughts = generate_thoughts(problem, thought_chain, n=BEAM_WIDTH)
            for thought in new_thoughts:
                extended = f"{thought_chain}\n{thought}".strip()
                solved, answer = is_solution(problem, extended)
                if solved:
                    return answer
                score = evaluate_thought(problem, extended)
                candidates.append((score, extended))

        # Zachowaj tylko BEAM_WIDTH najlepszych galezi
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [c for _, c in candidates[:BEAM_WIDTH]]

    # Jesli nie znaleziono rozwiazania — zwroc najlepszy tok mysli
    return beam[0] if beam else "Nie znaleziono rozwiazania w podanej glebokosci."


_PROBLEMS = [
    (
        "Produkcja: auth-service i payment-service sa niedostepne. api-gateway zwraca 503. "
        "auth-service log: 'DataSource exhausted, cannot acquire connection'. "
        "payment-service log: 'Timeout waiting for connection from pool'. "
        "PostgreSQL: 398/400 aktywnych polaczen. Redis: dziala (ping 2ms). "
        "Ruch uzytkownikow w normie. Zadnego deployu od 5 dni. "
        "Zaproponuj wszystkie mozliwe hipotezy, oceń je i wybierz optymalny plan przywrocenia."
    ),
    (
        "order-service: error rate 23% (norma: <0.5%). "
        "Bledy rozkladaja sie nastepujaco: 61% to HTTP 504 do payment-service, "
        "39% to HTTP 500 wewnetrzny z logiem 'inventory lock timeout after 5000ms'. "
        "payment-service dziala — jego wlasny error rate to 0.2%. "
        "inventory-service CPU: 91%, MEM: 78%. "
        "10 minut temu uruchomiono batch job 'nightly-stock-reconciliation'. "
        "Rozważ wszystkie mozliwe przyczyny i zaproponuj kolejnosc dzialan."
    ),
    (
        "Po deploymencie biblioteki logowania (log4j -> logback) o 03:15 "
        "CPU wszystkich 4 instancji api-gateway wzrosl z ~30% do 88-94%. "
        "Ruch: 1200 req/s (norma). Latencja p50: bez zmian (95ms). Latencja p99: wzrosla z 200ms do 1800ms. "
        "Nowe endpointy w tym deploymencie: brak. Zmiana: tylko biblioteka logowania + format logow JSON. "
        "Wolumen logow: wzrosl 4x (kazde zadanie HTTP loguje teraz pelne naglowki). "
        "Zaproponuj hipotezy, wskaż ktora jest najbardziej prawdopodobna i jak ja potwierdzic."
    ),
]


if __name__ == "__main__":
    import random
    problem = random.choice(_PROBLEMS)
    print("=== Tree of Thoughts (BFS) ===")
    print(f"Problem: {problem}\n")
    solution = tree_of_thoughts_bfs(problem)
    print(f"Rozwiazanie:\n{solution}")
