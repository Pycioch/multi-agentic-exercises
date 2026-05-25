"""
Pattern: Minimal LangChain Invocation (single call + print)
==========================================================
The simplest possible LangChain showcase in this directory:
- create one ChatOpenAI model
- send one prompt with invoke()
- print the returned text
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say hello in one short sentence for a workshop demo.")
    print(response.content)
