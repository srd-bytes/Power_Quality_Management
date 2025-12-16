from langgraph.graph import StateGraph  # Graph abstraction from LangGraph
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv  # Load GOOGLE_API_KEY and related secrets
from typing import TypedDict, List
import os
from report2pdf import get_report  # Helper to render markdown reports into PDFs

# Load environment variables
load_dotenv()
report_path = "reports"

# Define graph state passed between LangGraph nodes.
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
# Initialize embedding model and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")  # Shared embedding model
vectordb = Chroma(
    collection_name="IEEE",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db",
)

powerdb = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db"  # Optional: for persistence
)
systemmanual = Chroma(
    collection_name="power_system",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db",
)

# Load LLM
llm = GoogleGenerativeAI(model="gemini-2.5-flash")  # LLM backing the analysis/report agent

# Nodes
def retrieve_docs_node(state):
    query = state["query"]
    docs = vectordb.similarity_search(query, k=5)
    state["docs"] = docs
    return state

def getpowersystem(state):
    query = state["query"]
    docs = systemmanual.similarity_search(query, k=5)
    state["docs"] = docs
    return state

def summarize_node(state):
    """Summarize retrieved IEEE/power‑system context relative to the user query."""
    chunks = "\n\n".join([doc.page_content for doc in state["docs"]])
    prompt_template = PromptTemplate(
        input_variables=["prompt", "chunks"],
        template="""
        You are a summarizer node used in a RAG system. Summarize the best info for the user from the retrieved chunks, relevant to the prompt.

        chunks:
        {chunks}

        prompt:
        {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"], chunks=chunks)
    summary = llm.invoke(prompt_text)
    state["summary"] = summary
    return state

def analyze_node(state):
    """Analyze the summarized data and generate detailed technical insights."""
    summary_or_docs = state.get("summary") or "\n\n".join([doc.page_content for doc in state["docs"]])
    prompt_template = PromptTemplate(
        input_variables=["prompt", "data", "report_sample"],
        template="""
        You are an analyzer node. Your job is to analyze the power quality event data and give insights in the format below:

        Report Format:
        {report_sample}
        
        please generate report by using this power system info dont give general things.

        power_system:
        {{
        "model": "Simulink_14Bus_System",

        "buses": [
            "BUS1","BUS2","BUS3","BUS4","BUS5","BUS6","BUS7",
            "BUS8","BUS9","BUS10","BUS11","BUS12","BUS13","BUS14"
        ],

        "nodes": {{
            "BUS1": ["BUS1_A","BUS1_B","BUS1_C"],
            "BUS2": ["BUS2_A","BUS2_B","BUS2_C"],
            "BUS3": ["BUS3_A","BUS3_B","BUS3_C"],
            "BUS4": ["BUS4_A","BUS4_B","BUS4_C"],
            "BUS5": ["BUS5_A","BUS5_B","BUS5_C"],
            "BUS6": ["BUS6_A","BUS6_B","BUS6_C"],
            "BUS7": ["BUS7_A","BUS7_B","BUS7_C"],
            "BUS8": ["BUS8_A","BUS8_B","BUS8_C"],
            "BUS9": ["BUS9_A","BUS9_B","BUS9_C"],
            "BUS10": ["BUS10_A","BUS10_B","BUS10_C"],
            "BUS11": ["BUS11_A","BUS11_B","BUS11_C"],
            "BUS12": ["BUS12_A","BUS12_B","BUS12_C"],
            "BUS13": ["BUS13_A","BUS13_B","BUS13_C"],
            "BUS14": ["BUS14_A","BUS14_B","BUS14_C"]
        }},

        "generators": [
            {{ "name": "GEN1", "bus": "BUS2" }},
            {{ "name": "GEN2", "bus": "BUS3" }}
        ],

        "sync_compensators": [
            {{ "name": "SYNC_COMP1", "bus": "BUS3" }},
            {{ "name": "SYNC_COMP2", "bus": "BUS5" }},
            {{ "name": "SYNC_COMP3", "bus": "BUS6" }}
        ],

        "loads": [
            {{ "name": "LOAD4", "bus": "BUS4" }},
            {{ "name": "LOAD5", "bus": "BUS5" }},
            {{ "name": "LOAD7", "bus": "BUS7" }},
            {{ "name": "LOAD9", "bus": "BUS9" }},
            {{ "name": "LOAD10", "bus": "BUS10" }},
            {{ "name": "LOAD12", "bus": "BUS12" }},
            {{ "name": "LOAD13", "bus": "BUS13" }},
            {{ "name": "LOAD14", "bus": "BUS14" }}
        ],

        "lines": [
            {{ "name": "LINE1_2", "from": "BUS1", "to": "BUS2" }},
            {{ "name": "LINE2_3", "from": "BUS2", "to": "BUS3" }},
            {{ "name": "LINE2_4", "from": "BUS2", "to": "BUS4" }},
            {{ "name": "LINE3_4", "from": "BUS3", "to": "BUS4" }},
            {{ "name": "LINE4_5", "from": "BUS4", "to": "BUS5" }},
            {{ "name": "LINE5_6", "from": "BUS5", "to": "BUS6" }},
            {{ "name": "LINE6_7", "from": "BUS6", "to": "BUS7" }},
            {{ "name": "LINE7_9", "from": "BUS7", "to": "BUS9" }},
            {{ "name": "LINE9_10", "from": "BUS9", "to": "BUS10" }},
            {{ "name": "LINE10_11", "from": "BUS10", "to": "BUS11" }},
            {{ "name": "LINE11_12", "from": "BUS11", "to": "BUS12" }},
            {{ "name": "LINE12_13", "from": "BUS12", "to": "BUS13" }},
            {{ "name": "LINE13_14", "from": "BUS13", "to": "BUS14" }}
        ],

        "measurements": {{
            "voltage": {{
            "BUS1": "VM1","BUS2": "VM2","BUS3": "VM3","BUS4": "VM4",
            "BUS5": "VM5","BUS6": "VM6","BUS7": "VM7","BUS8": "VM8",
            "BUS9": "VM9","BUS10": "VM10","BUS11": "VM11","BUS12": "VM12",
            "BUS13": "VM13","BUS14": "VM14"
            }},
            "current": {{
            "BUS1": "IM1","BUS2": "IM2","BUS3": "IM3","BUS4": "IM4",
            "BUS5": "IM5","BUS6": "IM6","BUS7": "IM7","BUS8": "IM8",
            "BUS9": "IM9","BUS10": "IM10","BUS11": "IM11","BUS12": "IM12",
            "BUS13": "IM13","BUS14": "IM14"
            }}
        }},

        "rectifier": {{
            "ac_bus": "BUS11",
            "dc_nodes": ["DC_POS", "DC_NEG"],
            "components": ["DIODE_BRIDGE", "DC_LINK_CAP", "DC_LOAD", "RECTIFIER_MEAS"]
        }},

        "export_chain": {{
            "mux": "MUX_84",
            "output": "VEC84",
            "convert": "TO_SINGLE",
            "udp": "UDP_SEND"
        }},

        "powergui": {{
            "mode": "continuous",
            "domain": "three_phase"
        }}
        }}


        Retrieved Data:
        {data}

        Prompt:
        {prompt}
        """
    )
    with open("src/AI_agent/sample_texts/report_sample.txt", "r") as f:
        report_sample = f.read()

    prompt_text = prompt_template.format(prompt=state["query"], data=summary_or_docs, report_sample=report_sample)
    analysis = llm.invoke(prompt_text)
    state["analysis"] = analysis
    return state

def retrievebyattribute(state):
    all_metadata = powerdb.get(include=["metadatas"])["metadatas"]

    prompt_template = PromptTemplate(
        input_variables=["prompt","data"],
        template="""
        you are node to pretdict the event by time and various attribuite provided to you by the user
                
        retreived data = {data}

        user_query = {prompt}
        
        select the best event from the retrieved data and return in json format and dont return anything else.

        """
    
    )
    prompt_text = prompt_template.format(prompt=state["query"], data = all_metadata)
    best_instance = llm.invoke(prompt_text)
    state["best_instance"] = best_instance
    return state
 

def report_node(state):
    """Generate a final report markdown string using the analysis pipeline."""
    with open("src/AI_agent/sample_texts/report_sample.txt", "r") as f:
        report_sample = f.read()
    print("summarized",state["summary"])
    prompt_template = PromptTemplate(
        input_variables=["prompt", "data", "report_sample"],
        template="""
        You are a report generator node. Generate a report based on the retrieved data and prompt.

        Report Format:
        {report_sample}

        Retrieved Data:
        {data}

        Prompt:
        {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"], data=state['summary'], report_sample=report_sample)
    report = llm.invoke(prompt_text)
    state["report"] = report
    return state

def routing(state):
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
def chatbot(state):
    """Light‑weight chat node for general queries about the agent and system."""
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="""
        You are a chatbot made to answer casual queries for a Power Quality Management Agent.
        this agent can generate report,do chat,give info about the power system and analyze.
        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"])
    conv = llm.invoke(prompt_text)
    print(conv)
    state["query"] = ""
    state["conv"] = conv
    return state

# Router function
def router_node(state):
    query = state["node"]
    if "chatnode" in query.lower():
        return "chatnode"
    elif "report" in query.lower():
        return "retrieve_docs"
    elif "analyze" in query.lower():
        return "analyze"
    elif 'systeminfo' in query.lower():
        return "infosys"
        

def system_info(state):
    """Answer detailed questions about the modeled power system using retrieved docs."""
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="""
        You are a node in a Power Quality Management Agent your work is to use the fetched info and utlize this info with the use prompt to answer the query .
        mainly the queries realted to the poower system will be asked to you.

        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"])
    conv = llm.invoke(prompt_text)
    print(conv)
    state["query"] = ""
    state["conv"] = conv
    return state
builder = StateGraph(GraphState)  # Build and wire up the LangGraph

builder.add_node("chatnode", chatbot)
builder.add_node("router", routing)
builder.add_node("retrieve_docs", retrieve_docs_node)
builder.add_node("summarize", summarize_node)
builder.add_node("analyze", analyze_node)
builder.add_node("recommend", report_node)
builder.add_node("retrieve_docs_1", retrieve_docs_node)
builder.add_node("summarize_1", summarize_node)
builder.add_node("infosys",system_info)
builder.set_entry_point("router")

builder.add_conditional_edges("router", router_node, {
    "chatnode": "chatnode",
    "retrieve_docs": "retrieve_docs",
    "analyze":"retrieve_docs_1",
    'infosys':"infosys"
})
builder.add_edge("retrieve_docs_1", "summarize_1")
builder.add_edge("summarize_1", "analyze")

builder.add_edge("retrieve_docs", "summarize")
builder.add_edge("summarize", "recommend")  # Or "analyze" if you want analysis step
builder.add_edge('chatnode',"router")
builder.set_finish_point("recommend")
builder.set_finish_point("analyze")

graph = builder.compile()
output = graph.invoke({
    "query": ""
})
# output = graph.invoke({"query":""})
# print(output["conv"])
print(graph.get_graph().draw_mermaid())
print(output['report'])
get_report(output["report"], report_path)
