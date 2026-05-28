"""
Handoff miedzy agentami — OpenAI Agents SDK
Agent wsparcia przekazuje incydenty P0 do agenta eskalacji przez handoff().

Zrodlo: agent_communication/handoff.md
Slajd prezentacji: 7.15 — OpenAI Agents SDK handoff()
Docs: langchain_docs.md § ChatOpenAI (porownanie z LangChain/LangGraph podejsciem)

Uwaga: wymaga `openai-agents` (pip install openai-agents).
Pokazany jako porownanie z LangGraph Command — ten SDK jest zwiazany z modelami OpenAI
i nie ma wbudowanego checkpointingu dla dlugich przeplywow.
"""
from dotenv import load_dotenv
load_dotenv()

try:
    from agents import Agent, handoff, Runner
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Brak pakietu 'agents'. Zainstaluj zaleznosc: pip install openai-agents"
    ) from exc


escalation_agent = Agent(
    name="Escalation",
    instructions=(
        "Zajmujesz sie krytycznymi incydentami P0 wymagajacymi glebokiej analizy. "
        "Przeprowadz analise przyczyn i zaproponuj plan naprawczy."
    ),
    model="gpt-4o",
)

support_agent = Agent(
    name="Support",
    instructions=(
        "Odpowiadasz na ogolne pytania CloudOps. "
        "Dla incydentow P0 przekaz do agenta eskalacji."
    ),
    model="gpt-4o",
    tools=[
        handoff(
            escalation_agent,
            tool_name_override="escalate_to_specialist",
            tool_description_override="Przekaz incydenty P0 wymagajace glebokiej analizy",
        )
    ],
)


if __name__ == "__main__":
    result = Runner.run_sync(
        support_agent,
        "Incydent P0: auth-service nie odpowiada w DC-A od 15 minut.",
    )
    print(result.final_output)
