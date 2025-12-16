from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
import threading

load_dotenv()

_db = None
_embed = None
_lock = threading.Lock()


def get_db():
    """Lazy‑initialize and return the shared Chroma vector store used for fault events."""
    global _db, _embed
    if _db is None:
        with _lock:
            if _db is None:
                # 1. Initialize the Google Generative AI Embeddings model
                _embed = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004"
                )
                # 2. Initialize Chroma, passing the embedding function
                _db = Chroma(
                    collection_name="my_collection",
                    embedding_function=_embed,
                    persist_directory="chroma_langchain_db",
                )
    return _db


def upsert_ongoing_event(fault: dict):
    """
    FIXED: Uses the high-level db.add_documents method instead of 
           the low-level db._collection.upsert to ensure the 
           embedding function is correctly utilized.
    """
    db = get_db()
    event_id = f"{fault['location']}_{fault['time']}_ongoing"

    doc = Document( # Create a Document object
        page_content=(
            f"{fault['type']} ongoing at {fault['location']} "
            f"since {fault['time']}."
        ),
        metadata=fault,
    )
    
    # Using add_documents ensures the embedding function is called
    # and the vector is correctly created and added/updated (upserted).
    db.add_documents([doc], ids=[event_id])


def upload_to_database_done(fault: dict):
    # This function uses the high‑level API so embeddings are created automatically.
    db = get_db()

    doc = Document(
        page_content=(
            f"{fault['type']} occurred at {fault['location']} "
            f"on {fault['time']} lasting {fault['duration_ms']} ms."
        ),
        metadata=fault,
    )

    db.add_documents([doc], ids=[str(uuid4())])