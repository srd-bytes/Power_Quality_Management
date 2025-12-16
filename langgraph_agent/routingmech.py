from langgraph.graph import StateGraph  # Graph abstraction for routing flows
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv  # Load environment variables such as GOOGLE_API_KEY
from typing import TypedDict, List
import os
from report2pdf import get_report  # PDF report helper (not directly used here)

# Load environment variables for the LLM / embeddings.
load_dotenv()
report_path = "reports"

# Minimal graph state used by this routing‑only module.
class GraphState(TypedDict):
    query: str
    docs: List[Document]
    summary: str
    analysis: str
    recommendation: str
    report: str 
    node: str
    conv: str
    best_instance: str

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


llm = GoogleGenerativeAI(model="gemini-2.5-flash")  # LLM used to decide routing



def routing(state):
    """Route a user query to one of the high‑level nodes (chat, report, faults, etc.)."""
    if state["query"] =="":
        query = input()
    else:
        query = state["query"]
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="""
        This is a Power Quality Management Agent.
        You are a chat node which routes the user query to one of the following nodes:
        1) chatnode
        2) report
        3) prevfaults
        4) systeminfo
        5) analyze
        Just return the node name only.
        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=query)
    result = llm.invoke(prompt_text)
    print(f"router trigerred {result}")
    state["node"] = result
    state["query"] = query
    # {"node": result,"query":query}
    return state