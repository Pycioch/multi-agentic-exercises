"""
Wiazanie uzytkownika przez HMAC(dzierzawca + uzytkownik + sesja)
Kryptograficznie bezpieczny thread_id chroniacy przed atakami IDOR.

Zrodlo: agentops/bezpieczenstwo_systemow.md (sekcja User Binding)
Slajd prezentacji: 9.8 — Powiazanie uzytkownika przez HMAC
Docs: langchain_docs.md § create_agent — checkpointer + thread_id
"""
from dotenv import load_dotenv
load_dotenv()

import hmac
import hashlib
import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver


def make_thread_id(
    tenant_id: str, user_id: str, session_id: str, secret: bytes
) -> str:
    """Generuje nieprzewidywalny thread_id — atakujacy bez klucza nie odtworzy go."""
    payload = f"{tenant_id}:{user_id}:{session_id}".encode()
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


@tool
def get_incidents(datacenter: str) -> str:
    """Zwraca liste incydentow P0 dla podanego datacenter."""
    return f"[SYMULACJA] 3 incydenty P0 w {datacenter}: INC-2040, INC-2041, INC-2042"


memory = MemorySaver()
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[get_incidents],
    system_prompt="Jestes agentem Cloud-Ops. Odpowiadaj zwiezle po polsku.",
    checkpointer=memory,
)


if __name__ == "__main__":
    SECRET = os.environ.get("THREAD_SECRET", "dev-secret").encode()

    # Kazdy uzytkownik otrzymuje unikalny, nieprzewidywalny thread_id
    thread_id = make_thread_id(
        tenant_id="intel-corp",
        user_id="alice@intel.com",
        session_id="sess-2025-abc",
        secret=SECRET,
    )
    print(f"thread_id: {thread_id[:16]}...  (deterministyczny dla tej sesji)")

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [("user", "Pokaz incydenty P0 z DC-A")]},
        config=config,
    )
    print(f"Odpowiedz: {result['messages'][-1].content}")

    # Weryfikacja: ten sam wejscie -> ten sam thread_id
    same_id = make_thread_id("intel-corp", "alice@intel.com", "sess-2025-abc", SECRET)
    assert same_id == thread_id, "thread_id musi byc deterministyczny"
    print("OK: thread_id deterministyczny dla tej samej trojki tenant+user+session")
