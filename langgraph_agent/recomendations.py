import langchain.prompts  # Prompt templates for the recommendation LLM
from langchain_chroma import Chroma  # Not used directly here but kept for compatibility
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI  # LLM / embeddings backend
from dotenv import load_dotenv  # Ensure API keys are loaded
import langchain  # Base LangChain package
from datetime import datetime
import yaml
import time
import os
load_dotenv()
import sys
import subprocess
llm = GoogleGenerativeAI(
            model = "gemini-2.5-flash"
        )

prompt_template = langchain.prompts.PromptTemplate(
    input_variables=["fault"],
    template="""
You are given a fault description. Your job is to:
- Provide recommendations related to the fault
- List possible causes
- Present fault details at the top
- Include IEEE references and nominal ranges (if applicable)
- Consider the provided power system JSON when generating insights
- Describe possible damages in the context of a power system
- Output only Markdown format

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

Fault input:
{fault}
please have time and type in a format such that regex can extract it
r"\*\*Time:\*\*\s*(.+)"
r"\*\*Type:\*\*\s*(.+)"
"""
)


faultfiles = [i.split(".txt")[0] for i in os.listdir("faults")]  # All detected fault snapshots
mdfiles = [i.split(".md")[0] for i in os.listdir("ai_recommendations")]  # Already‑generated recommendations

for i in faultfiles:

    if i in mdfiles:
        continue


    with open(f'faults/{i}.txt', 'r') as file:
        content = file.read()  

    prompt = prompt_template.format(fault =content )
    response = llm.invoke(prompt)
    response = response.strip().removeprefix("```md").removesuffix("```").strip()

    with open(f"ai_recommendations/{i.split('.txt')[0]}.md","w") as file:
        file.writelines(response)
    time.sleep(2)