# Code Showcases — Intel Agentic Systems

Krotkie, samodzielne pliki .py demonstrujace konkretne wzorce agentowe.
Kazdy plik uruchamia sie niezaleznie. Brak zaleznosci miedzy plikami.

Showcases NIE SA krokami progresji kodowej (kroki 1-10 z planu szkolenia).
To izolowane demonstracje wzorca — do pokazania na zywo obok slajdu.

## Pliki z slajdow prezentacji

| Plik | Wzorzec | Slajd |
|------|---------|-------|
| `01_structured_output_pydantic.py` | Pydantic + Instructor — walidacja schematu | 3.11 |
| `02_hitl_interrupt.py` | Human-in-the-Loop z `interrupt()` | 3.14 |
| `03_langgraph_command_handoff.py` | Handoff: LangGraph `Command(goto=...)` | 7.14 |
| `04_openai_sdk_handoff.py` | Handoff: OpenAI Agents SDK `handoff()` | 7.15 |
| `05_hmac_user_binding.py` | Wiazanie uzytkownika HMAC(tenant+user+session) | 9.8 |

## Pliki wzorcow agentowych (pattern_examples)

| Plik | Wzorzec | Modul warsztatowy |
|------|---------|-------------------|
| `06_react.py` | ReAct: Thought -> Action -> Observation | M3 |
| `07_reflection_reflact.py` | Reflection / ReflAct (+27.7% nad ReAct) | M4, M5 |
| `08_orchestrator_worker.py` | Orchestrator-Worker z Send() fan-out | M5 |
| `09_planning_plan_execute.py` | Plan-and-Execute z replanning | M5 |
| `10_tree_of_thoughts.py` | Tree of Thoughts BFS (kosztowny — patrz uwaga) | M5 |

## Setup

Skopiuj `.env.example` do `.env` i wpisz klucz API:
```
cp .env.example .env
# edytuj .env: OPENAI_API_KEY=sk-...
```

Zaleznosci:
```
pip install langchain langchain-openai langgraph pydantic python-dotenv
# 04: pip install openai-agents  (wymaga konta OpenAI)
```

## Zrodla

Kod oparty na materialach badawczych:
- `theory_research/design_patterns/` — react_got.md, goal_state_reflection.md, anthropic_multi_agent_blueprint.md, guide_to_agentic_design_patterns.md
- `theory_research/agentops/` — durable_execution_pattern.md, bezpieczenstwo_systemow.md
- `theory_research/agent_communication/` — handoff.md, mcp_a2a_acp_ucp.md
