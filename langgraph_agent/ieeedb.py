from dotenv import load_dotenv,dotenv_values,set_key  # Utilities for managing .env and credentials
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
import base64
from uuid import uuid4
import email
from datetime import datetime
import time
import re

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"  # Embedding model used for IEEE knowledge base
)
ieee_sample_docs = [

  {
    "type": "Voltage Swell",
    "standard": "IEEE 1159",
    "metrics": ["Duration", "Peak Amplitude (pu or %)"],
    "thresholds": {
      "amplitude": "1.1 to 1.8 pu (110% to 180% of nominal voltage)",
      "duration": "0.5 cycles to 1 minute"
    }
  },
  {
    "type": "Voltage Sag (Dip)",
    "standard": "IEEE 1159, IEC 61000-4-11",
    "metrics": ["Duration", "Minimum Voltage Level"],
    "thresholds": {
      "amplitude": "0.1 to 0.9 pu (10% to 90% of nominal voltage)",
      "duration": "0.5 cycles to 1 minute"
    }
  },
  {
    "type": "Voltage Interruption",
    "standard": "IEEE 1159",
    "metrics": ["Duration", "Remaining Voltage"],
    "thresholds": {
      "voltage": "< 0.1 pu",
      "categories": {
        "momentary": "< 3 seconds",
        "temporary": "3 seconds to 1 minute",
        "sustained": "> 1 minute"
      }
    }
  },
  {
    "type": "Voltage Unbalance",
    "standard": "IEEE 1159, IEC 61000-4-30",
    "metrics": ["Voltage Unbalance (%)"],
    "thresholds": {
      "max_unbalance": "≤ 2% in balanced 3-phase systems"
    }
  },
  {
    "type": "Harmonics",
    "standard": "IEEE 519-2014",
    "metrics": ["Total Harmonic Distortion (THD)", "Individual Harmonics"],
    "thresholds": {
      "voltage_thd": "≤ 5%",
      "current_thd": "≤ 8% (for low-voltage systems)"
    }
  },
  {
    "type": "Notching",
    "standard": "IEEE 519, IEC 61000-4-7",
    "metrics": ["Notch Depth", "Notch Width"],
    "thresholds": {
      "impact": "Should not affect zero crossing or control timing"
    }
  },
  {
    "type": "Flicker",
    "standard": "IEC 61000-4-15",
    "metrics": ["Pst", "Plt"],
    "thresholds": {
      "Pst": "≤ 1.0",
      "Plt": "≤ 0.8"
    }
  },
  {
    "type": "Frequency Deviation",
    "standard": "IEEE 1159",
    "metrics": ["Measured Frequency (Hz)"],
    "thresholds": {
      "deviation": "±0.5 Hz (typical), ±1 Hz (max allowable)"
    }
  },
  {
    "type": "Transient Overvoltage (Impulsive)",
    "standard": "IEEE 1159, IEC 61000-4-5",
    "metrics": ["Peak Voltage", "Rise Time", "Duration"],
    "thresholds": {
      "peak_voltage": "Can exceed several kV",
      "rise_time": "< 1 μs",
      "duration": "< 1 ms"
    }
  },
  {
    "type": "Short Circuit Event",
    "standard": "IEEE C37.011",
    "metrics": ["Fault Current (kA)", "Duration", "Impedance"],
    "thresholds": {
      "fault_current": "5–50 kA typical in industrial systems",
      "duration": "Depends on protection scheme"
    }
  }
]
vectordb = Chroma(
    collection_name="IEEE",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db",
)

docs = []
for text in ieee_sample_docs:
    doc = Document(
        page_content=str(text),
    )
    docs.append(doc)

uuids = [str(uuid4()) for _ in docs]
vectordb.add_documents(documents=docs, document_id=uuids)