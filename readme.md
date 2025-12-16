## Power Quality Management – Project README

This project is a **Power Quality Management** system that:
- Streams power system measurements over UDP into shared memory.
- Detects voltage sag/swell events in real time.
- Logs events to disk and a vector database.
- Serves a web dashboard and AI assistant for analysis and reporting.

The main web app lives in `app.py`, and the AI/RAG agent stack is under `langgraph_agent/`.

---

## 1. Environment Setup

- **Python version**: 3.10+ recommended  
- **Install dependencies**:

```bash
cd /home/vc940/Work/eee
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

- **Environment variables**:
  - Create a `.env` file at the project root with your Google Generative AI credentials, e.g.:

```bash
GOOGLE_API_KEY="your_api_key_here"
```

You may need additional keys depending on how `langchain_google_genai` is configured in your environment.

---

## 2. Data & Vector DB Preparation

This project uses **ChromaDB** for vector storage in the `chroma_langchain_db/` directory.

- Ensure the `chroma_langchain_db/` folder is present (it should already be in this repo).
- Make sure your IEEE and power system documents are loaded into Chroma via the utilities in `langgraph_agent/database.py` / `langgraph_agent/ieeedb.py` (those scripts are intended to build and update the collections).

Fault events and AI recommendations are stored under:
- `faults/`
- `ai_recommendations/`
- Generated reports (PDF) are stored in `reports/`.

---

## 3. Running the Web Application

The main entry point is `app.py`. It:
- Listens for UDP power-quality data.
- Detects events and writes them to shared memory & disk.
- Starts a Flask server that exposes REST endpoints and a web UI.

### 3.1 Start the Flask Server (Development)

From the project root:

```bash
source .venv/bin/activate
python app.py
```

By default the server runs on `http://0.0.0.0:5000`.

### 3.2 Using Gunicorn/Uvicorn (Production-style)

You can also run it with Gunicorn:

```bash
source .venv/bin/activate
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

> Note: If you run with Gunicorn, the UDP/shared-memory logger started in `if __name__ == "__main__":` is not launched automatically. For full end‑to‑end real‑time streaming, use `python app.py` or adapt the process management to your deployment.

---

## 4. Frontend & API Endpoints

The Flask app serves:
- `GET /` – Main dashboard (`templates/index.html`).
- `GET /waveform_data` – Latest waveform chunk (for `static/waveform.js`).
- `GET /reports` – List of generated PDF reports in `reports/`.
- `GET /reports/<filename>` – Download a specific report.
- `GET /faults` – List fault event JSON files in `faults/`.
- `GET /faults/<filename>` – Get a specific fault file.
- `GET /list_recommendations` – List AI recommendation markdown files in `ai_recommendations/`.
- `GET /recommendations/<filename>` – Fetch the content of a recommendation.
- `POST /generate_recommendation` – Triggers `langgraph_agent/recomendations.py` to generate a new recommendation.
- `POST /chat` – Sends a chat message to the LangGraph AI agent and returns an HTML-formatted response and updated history.

These endpoints are consumed by the JavaScript files in `static/` and the HTML templates in `templates/`.

---

## 5. LangGraph / LangChain AI Agent

The AI agent is defined primarily in:
- `langgraph_agent/main.py` – Builds the LangGraph state machine.
- `langgraph_agent/agent.py` – Wraps the LangGraph graph into a callable `run()` used by `app.py`.
- `langgraph_agent/recomendations.py` – Generates AI-based recommendations and saves them into `ai_recommendations/`.
- `langgraph_agent/report2pdf.py` – Converts markdown reports into PDFs and saves them in `reports/`.

### 5.1 How the Agent is Used from the Web App

- `app.py` imports `langgraph_agent.agent.run` as `langgraph_run`.
- The `/chat` route calls `langgraph_run(message, history)` and:
  - Logs the interaction into the vector DB queue.
  - Returns both raw and HTML‑formatted content to the frontend.

### 5.2 Running the Agent Standalone

You can also invoke the agent from the command line for debugging:

```bash
source .venv/bin/activate
python -m langgraph_agent.main
```

This will execute the LangGraph pipeline as defined in `main.py`, generate a report, and render it to PDF using `report2pdf.py`.

---

## 6. UDP Data Streaming & Fault Detection

The real‑time data path is implemented in `app.py`:
- `logger_process()` – Reads UDP packets, writes them into two shared memory buffers (`shmA`, `shmB`).
- `bridge_thread()` – Pulls buffers from shared memory, runs `detect_event()`, and:
  - Updates `latest_data_chunk` for the waveform endpoint.
  - Handles event start/stop logic (sag/swell).
  - Writes fault metadata to `faults/<timestamp>.txt`.
  - Sends fault records to the DB worker via `db_queue`.

To feed the system with data you need a sender that sends UDP packets to:
- Host: `127.0.0.1`
- Port: `25000`
- Payload: 84 floats (`struct` format `<84f>`), one sample per packet.

---

## 7. Fault Model (PyTorch)

The file `fault_detection/fault_model.py` contains a PyTorch model for fault detection/visualization (e.g., with `torchview` and `torchsummary`). This is not wired directly into `app.py` yet, but:
- You can extend the real‑time pipeline to run model inference on each batch.
- Or use it offline on logged data from `bus_data_batch_*.xlsx`.

---

## 8. Project Structure (High-Level)

- `app.py` – Main Flask app, shared memory, UDP logger, event detection, REST endpoints.
- `langgraph_agent/` – LangGraph/LangChain RAG agent, report generation, vector DB utilities.
- `fault_detection/` – PyTorch fault detection model.
- `templates/` – HTML templates for the web UI.
- `static/` – Frontend JS for chat and waveform visualization.
- `documents/` – Reference/IEEE/Power quality PDFs used to build the vector DB.
- `ai_recommendations/` – AI-generated recommendations as markdown.
- `faults/` – Serialized fault events as JSON-like `.txt` files.
- `reports/` – Generated PDF reports.
- `chroma_langchain_db/` – Chroma vector DB storage.

---

## 9. Common Commands

```bash
# 1) Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate

# 2) Install all dependencies
pip install -r requirements.txt

# 3) Run the web app
python app.py

# 4) (Optional) Generate a new recommendation via backend
curl -X POST http://localhost:5000/generate_recommendation

# 5) (Optional) Test chat endpoint
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the last sag event", "history": []}'
```

This README should give you everything needed to **set up, run, and extend** the Power Quality Management system.
