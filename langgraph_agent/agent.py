from langgraph.graph import StateGraph  # Orchestrates the conversational graph
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv  # Load GOOGLE_API_KEY and other secrets
from typing import TypedDict, List
import os
# from report2pdf import get_report
import time
import markdown2  # Markdown → HTML converter for report generation
from weasyprint import HTML  # HTML → PDF rendering
from pathlib import Path
from markdown import markdown  # Not used directly here, but kept for compatibility


def get_report(md_string, path):
    """Render a markdown report string to a timestamped PDF file."""
    time_ = time.time()
    html = markdown2.markdown(md_string, extras=["tables", "fenced-code-blocks"])
    css = """
    <html><head><style>
    body { font-family: Arial; font-size: 10pt; margin: 0.5cm; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { border: 1px solid #aaa; padding: 4px; text-align: left; }
    h1, h2, h3 { margin: 6px 0; }
    img { max-width: 250px; max_height:125px ;height: auto; }
    </style></head><body>
    """ + html + "</body></html>"

    HTML(string=css).write_pdf(f"{path}/{time_}.pdf")


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
    previous:str
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


llm = GoogleGenerativeAI(model="gemini-2.5-flash")  # Main LLM backing all nodes
def chatbot(state):
    """General chat node that answers casual queries and updates conversation history."""
    a = time.time()
    prompt_template = PromptTemplate(
        input_variables=["prompt","prev"],
        template="""
        You are a chatbot made to answer casual queries for a Power Quality Management Agent.
        this agent can generate report,do chat,give info about the power system and analyze. if require use previous conversations to get more context.

        Previous_text = {prev}
        
        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"],prev = state["previous"][:-1000])
    time.sleep(0.25)
    conv = llm.invoke(prompt_text)
    print("Agent:",conv)
    state["conv"] +=  f"{conv}\n"
    state["previous"][:-1000] +=f"user:{state['query']}\n Agent:{conv}\n"
    state["query"] = ""
    print(f"Chatbot took {time.time()-a:.2f} seconds")
    return state

def powerchat_(state):
    """Domain‑specific chat node that focuses on power‑system related queries."""
    a = time.time()
    prompt_template = PromptTemplate(
        input_variables=["prompt","prev",'fault'],
        template="""
        you will be just needed to analyze the previous conversations and answer the queries related to power system regarding the power system user ask question give him recomendations according to ieee standars and try to resolve his queries
        Previous_text = {prev}
        
        data of fault  = {fault}

        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"],prev = state["previous"][:-1000],fault = state['best_instance'])
    time.sleep(0.25)
    conv = llm.invoke(prompt_text)
    print("Agent:",conv,"Do want to say something else.")
    
    state["conv"] +=  f"{conv}\n"
    state["previous"][:-1000] +=f"user:{state['query']}\n Agent:{conv}\n"
    print(f"Power chat took {time.time()-a:.2f} seconds")
    return state

def retrievebyattribute(state):
    """Retrieve candidate fault events from powerdb based on user query and history."""
    a = time.time()
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
    prompt_text = prompt_template.format(prompt=state["query"], data = all_metadata,previous_chats = state["previous"][:-1000])
    time.sleep(0.25)
    best_instance = llm.invoke(prompt_text)
    state["best_instance"] = best_instance

    print(f"retrieval from power system took {time.time()-a:.2f} seconds")
    print("retrieved event:",best_instance)
    return state


def retrieve_docs_node(state):
    query = state["query"]
    docs = vectordb.similarity_search(query, k=5)
    state["docs"] = docs
    return state

def reportagent(state):
    """Generate a structured report using IEEE docs, selected event and conversation context."""
    a = time.time()
    with open("langgraph_agent/sample_texts/report_sample.txt", "r") as f:
        report_sample = f.read()
    # print(state["docs"])
    if "null" in state["best_instance"].lower():
        print("No data found")
        return state
    print(state["docs"],state["best_instance"])
    prompt_template = PromptTemplate(
        input_variables=["prompt", "data", "report_sample","eventdata","prev_chats"],
        template="""
        You are a report generator node. Generate a report based on the Ieee retrieved data,event and prompt. if needed use the previous chats to retain the context


        Report Format:
        {report_sample}

        and please give generate the report according to this power system schematics dont give general suggestions.

                
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

        Event:
        {eventdata}

        Previous chats:
        {prev_chats}
        
        Prompt:
        {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"], data=state["docs"],eventdata = state["best_instance"] ,report_sample=report_sample,prev_chats = state["previous"][:-1000])
    time.sleep(0.25)
    report = llm.invoke(prompt_text)
    state["report"] = report
    get_report(state["report"], report_path)
    print("Agent:report has been prepared succesfully thanks for using our services,is there anything else you want to do?")
    state["conv"]+= "report has been prepared succesfully thanks for using our services,is there anything else you want to do?"
    state["query"] = ""

    state["previous"][:-1000] +=f"user:{state['query']}\n Agent:report has been prepared succesfully thanks for using our services,is there anything else you want to do?\n"
    print(f"Report generation took {time.time()-a:.2f} seconds")
    return state


def routing(state):
    """Top‑level router: decide between generic chat, power system flow, or exit."""
    a = time.time()
    # if state["query"] =="":
    #     query = input("user:")
    # else:
    query = state["query"]
    prompt_template = PromptTemplate(
        input_variables=["prompt","prev_conv"],
        template="""
        This is a Power Quality Management Agent.
        use previous conversations to get the context if needed.
        You are a chat node which routes the user query to one of the following nodes:
        1) chatnode
        2) power_system (if user asks info about any of power system components or tell to generate report)
        3)exit
        if intent is unclear direct towards the chat and will try to get more details
        Just return the node name only.
        
        previous chats :{prev_conv}
        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=query,prev_conv = state["conv"])
    time.sleep(0.25)
    result = llm.invoke(prompt_text)
    print(f"router trigerred {result}")
    state["node"] = result
    state["query"] = query
    # {"node": result,"query":query}
    print(f"Routing took {time.time()-a:.2f} seconds")
    return state


def routingpower(state):
    """Second‑level router used once the user is in the power‑system flow."""
    a = time.time()
    prompt_template = PromptTemplate(
            input_variables=["prompt","prev_conv"],
            template="""
            This is a Power Quality Management Agent.
            use previous conversations to get the context if needed.
            You are a chat node which routes the user query to one of the following nodes:
            1) chatrelatedtopowersystem (if user needs recomendation or preventive measures or ask about faults ex: no of times,where the fault ocured,deviation values,like that)
            2) report (only if report word is there in prompt)
            3) exit
            please give report only if report word is there in prompt else gice chatrelated to pwer system.
            if intent is unclear direct towards the chat and will try to get more details
            Just return the node name only.
            
            previous chats :{prev_conv}
            Prompt: {prompt}
            """
    )
    prompt_text = prompt_template.format(prompt=state["query"],prev_conv = state["previous"][:-1000])
    time.sleep(0.25)
    result = llm.invoke(prompt_text)
    
    print(f"power router trigerred {result}")
    state["node"] = result
     # {"node": result,"query":query}
    print(f"Power Routing took {time.time()-a:.2f} seconds")
    return state

def router_node(state):
    query = state["node"]
    if "chatnode" in query.lower():
        return "chatnode"
    elif "power_system" in query.lower():
        return "retriever"
    elif "exit" in query.lower():
        return "exit"

def router_node2(state):
    query = state["node"]

    if "report" in query.lower():
        return "retriever"
    elif "exit" in query.lower():
        return "exit"
    elif "chatrelatedtopowersystem" in query.lower():
        return "powerchat"
    else:
        print(f"thrown{query}")
        return "powerchat"
def exitnode(state):
    """Exit node that synthesizes a short farewell message based on previous chats."""
    a = time.time()
    prompt_template = PromptTemplate(
        input_variables=["prev_conv"],
        template="""
        you are exitnode of our agent give short exit greetings to the user by seeing the previous conversation. 

        previous chats :{prev_conv}
        just return the greetings ok which are ready to be used in application.
        """
    )
    prompt = prompt_template.format_prompt(prev_conv = state["previous"][:-1000][:-1000])
    time.sleep(0.25)
    print(llm.invoke(prompt))
    print(f"Exit node took {time.time()-a:.2f} seconds")
    return state
builder = StateGraph(GraphState)
builder.add_node("routing_power",routingpower)
builder.add_node("router",routing)
# builder.add_node("routernode",router_node)
builder.add_node("chatnode",chatbot)
builder.add_node("powerinfoi", retrievebyattribute)
builder.add_node("reportnode",reportagent)
builder.add_node("retriever",retrieve_docs_node)
builder.add_node("exit_node",exitnode)
builder.add_node("chat_power",powerchat_)
# builder.add_edge("powerinfo","retriever")
builder.add_edge("retriever","reportnode")
builder.add_edge("powerinfoi","routing_power")
builder.add_edge("reportnode","exit_node")
builder.add_edge("chat_power","exit_node")
builder.set_entry_point("router")
builder.set_finish_point("exit_node")
builder.add_edge("chatnode","exit_node")
builder.add_conditional_edges("router", router_node, {
    "chatnode": "chatnode",
    "retriever": "powerinfoi",
    "exit": "exit_node"
})

builder.add_conditional_edges("routing_power", router_node2, {
    "powerchat": "chat_power",
    "retriever": "retriever",
    "exit":"exit_node"
    })

graph = builder.compile()  # Compile the LangGraph into an executable graph object
print(graph.get_graph().draw_mermaid())
# output = graph.invoke({
#     "query": "","conv":""
# })
def run(message: str,previous ) -> str:
    """Public entry point used by `app.py` to run a single turn through the graph."""
    try:
        output = graph.invoke({
            "query": message,
            "conv": "",
            "previous":previous
        })
        full_conv = output.get("conv", "")
        return full_conv
        # Extract the latest full answer (multiline)
        lines = full_conv.strip().split("\n")
        answer_lines = []
        capture = False

        for line in reversed(lines):
            if line.strip().startswith("answer:"):
                answer_lines.append(line.replace("answer:", "").strip())
                capture = True
            elif capture and line.strip().startswith("query:"):
                break
            elif capture:
                answer_lines.append(line)

        # Reverse to maintain the original order
        final_answer = "\n".join(reversed(answer_lines)).strip()
        return final_answer if final_answer else "I'm not sure how to respond yet."

    except Exception as e:
        return f"Error: {str(e)}"
