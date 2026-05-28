"""
Structured output: Pydantic + with_structured_output()
Walidacja schematu wymuszana przez LangChain — nie prompt-engineering.

Zrodlo: design_patterns/structured_output.md, design_patterns/tool_use_function_calling.md
Slajd prezentacji: 3.11 — Instructor + Pydantic
Docs: langchain_docs.md § ChatOpenAI.with_structured_output()
"""
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from langchain_openai import ChatOpenAI


class IncidentSummary(BaseModel):
    incident_id: str
    severity: str = Field(description="P0, P1 lub P2")
    root_cause: Optional[str] = None
    resolution_time_minutes: int = Field(ge=0)
    affected_services: list[str]

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        if v not in ("P0", "P1", "P2"):
            raise ValueError(f"Nieznana wartosc severity: {v}")
        return v


def extract_incident(raw_text: str, structured_output: bool = False) -> IncidentSummary:
    llm = ChatOpenAI(model="gpt-5-mini")
    system_prompt = "Wyodrebnij dane incydentu i odpowiadaj po polsku."
    user_prompt = f"Przeanalizuj incydent i zwroc podsumowanie:\n\n{raw_text}"
    # with_structured_output() mapuje schemat Pydantic na native JSON schema OpenAI.
    # Walidacja semantyczna (field_validator) to nadal zadanie dewelopera.
    
    if structured_output:
        structured_llm = llm.with_structured_output(IncidentSummary)
        return structured_llm.invoke(raw_text)


    return llm.invoke([
        ("system", system_prompt),
        ("user", user_prompt),
    ])


if __name__ == "__main__":
    sample = """
    Przykładowy incydent: 
    INC-1023:  P0. 
    Wyczerpana pula polaczen do bazy danych na auth-service i payment-service. 
    Przyczyna: brak limitu polaczen w konfiguracji.
    Solution 12 minut. 
    Affected services: auth-service, payment-service.
    """

    structured_output = True
    result = extract_incident(sample, structured_output)

    if structured_output:
        print(result)
    else:
        print(result.content)
    # Wynik jest w pelni zwalidowanym obiektem Pydantic — nie stringiem, nie slownikiem.
