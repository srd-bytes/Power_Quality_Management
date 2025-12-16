from langgraph.graph import StateGraph  # Graph abstraction for orchestrating agent states
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv  # Load environment variables for the LLM backend
from typing import TypedDict, List
import os
from report2pdf import get_report  # Render markdown reports into PDFs
import time
load_dotenv()
report_path = "reports"
a = time.time()
# Define graph state
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
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")  # Shared embedding model
vectordb = Chroma(
    collection_name="IEEE",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db",
)

powerdb = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db"  
)


llm = GoogleGenerativeAI(model="gemini-2.5-flash")  # LLM that powers all nodes
def chatbot(state):
    """General chat node used in the simpler agent prototype."""
    prompt_template = PromptTemplate(
        input_variables=["prompt","prev"],
        template="""
        You are a chatbot made to answer casual queries for a Power Quality Management Agent.
        this agent can generate report,do chat,give info about the power system and analyze. if require use previous conversations to get more context.

        Previous_text = {prev}
        
        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"],prev = state["conv"])
    conv = llm.invoke(prompt_text)
    print(conv)
    state["conv"] +=  f"query:{state['query']},\n answer:{conv}\n"
    state["query"] = ""
    
    return state


def retrievebyattribute(state):
    """Pick the best matching event from `powerdb` given the user query."""
    all_metadata = powerdb.get(include=["metadatas"])["metadatas"]
    # print(all_metadata)
    prompt_template = PromptTemplate(
        input_variables=["prompt","data","previous_chats"],
        template="""
        you are node to pretdict the event by time and location to you by the user and you can also use the previous chats to analyze the context if they are relevant
                
        retreived data = {data}

        previous_conversations = {previous_chats}

        user_query = {prompt}
        
        select the best event from the retrieved data and return in json format and dont return anything else.

        if the exact thing is not found return none

        """
    
    )
    prompt_text = prompt_template.format(prompt=state["query"], data = all_metadata,previous_chats = "dafsads")
    best_instance = llm.invoke(prompt_text)
    state["best_instance"] = best_instance
    print("bestinstance",best_instance)
    return state


def retrieve_docs_node(state):
    query = state["query"]
    docs = vectordb.similarity_search(query, k=1)
    state["docs"] = docs
    return state

def reportgeneration(state):
    """Generate a summary report and write it to disk as a PDF."""
    with open("src/AI_agent/sample_texts/report_sample.txt", "r") as f:
        report_sample = f.read()
    # print(state["docs"])
    if "null" in state["best_instance"].lower():
        print("No data found")
        return state

    prompt_template = PromptTemplate(
        input_variables=["prompt", "data", "report_sample","eventdata","prev_chats"],
        template="""
        You are a report generator node. Generate a report based on the Ieee retrieved data,event and prompt. if needed use the previous chats to retain the context

        Report Format:
        {report_sample}

        Retrieved Data:
        {data}

        Event:
        {eventdata}

        Previous chats:
        {prev_chats}
        
        Prompt:
        {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"], data=state["docs"],eventdata = state["best_instance"] ,report_sample=report_sample,prev_chats = state["conv"])
    report = llm.invoke(prompt_text)
    state["report"] = report
    get_report(state["report"], report_path)
    print("report has been prepared succesfully thanks for using our services,is there anything else you want to do?")
    state["conv"]+= "answer:report has been prepared succesfully thanks for using our services,is there anything else you want to do?"
    state["query"] = ""
    return state


def routing(state):
    """Route user query to either chat, report flow, or exit."""
    if state["query"] =="":
        query = input()
    else:
        query = state["query"]
    prompt_template = PromptTemplate(
        input_variables=["prompt","prev_conv"],
        template="""
        This is a Power Quality Management Agent.
        use previous conversations to get the context if needed.
        You are a chat node which routes the user query to one of the following nodes:
        1) chatnode
        2) report
        3) exit
        if intent is unclear direct towards the chat and will try to get more details
        Just return the node name only.
        
        previous chats :{prev_conv}
        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=query,prev_conv = state["conv"])
    result = llm.invoke(prompt_text)
    print(f"router trigerred {result}")
    state["node"] = result
    state["query"] = query
    # {"node": result,"query":query}
    return state

def router_node(state):
    query = state["node"]
    if "chatnode" in query.lower():
        return "chatnode"
    elif "report" in query.lower():
        return "powerinfo"
    elif "exit" in query.lower():
        return "exit"
def exitnode(state):
    """Exit node that creates a short farewell message from conversation history."""
    prompt_template = PromptTemplate(
        input_variables=["prev_conv"],
        template="""
        you are exitnode of our agent give short exit greetings to the user by seeing the previous conversation. 

        previous chats :{prev_conv}
        just return the greetings ok which are ready to be used in application.
        """
    )
    prompt = prompt_template.format_prompt(prev_conv = state["conv"])
    print(llm.invoke(prompt))
    return state

builder = StateGraph(GraphState)  # Build the graph wiring between nodes
builder.add_node("router",routing)
builder.add_node("routernode",router_node)
builder.add_node("chatnode",chatbot)
builder.add_node("powerinfo", retrievebyattribute)
builder.add_node("reportnode",reportgeneration)
builder.add_node("retriever",retrieve_docs_node)
builder.add_node("exit_node",exitnode)
builder.add_edge("powerinfo","retriever")
builder.add_edge("retriever","reportnode")
builder.add_edge("reportnode","router")
builder.set_entry_point("router")
builder.set_finish_point("exit_node")
builder.add_edge("chatnode","router")
builder.add_conditional_edges("router", router_node, {
    "chatnode": "chatnode",
    "powerinfo": "powerinfo",
    "exit": "exit_node"
})
graph = builder.compile()
print(graph.get_graph().draw_mermaid())
output = graph.invoke({
    "query": "","conv":""
})


print(time.time()-a)