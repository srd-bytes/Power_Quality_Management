import socket  # UDP socket for incoming power-quality data
import struct
import numpy as np  # Numerical operations and shared-memory buffers
import time
import threading  # Used for bridge thread and DB worker
from queue import Queue
from multiprocessing import Process, Event  # Separate process for UDP logger
from multiprocessing.shared_memory import SharedMemory  # Shared buffers for streaming data
from flask import Flask, jsonify, request, render_template, send_from_directory  # HTTP API and templates
from flask_cors import CORS  # Allow cross-origin requests from frontend
import markdown  # Render chat responses as HTML
import os
import json
import re
import subprocess
import pandas as pd
from collections import defaultdict
import sys
import subprocess
# === Import Database Module ===
from langgraph_agent import database as vecdb

# === Import Agent (with Fallback) ===
try:
    from langgraph_agent.agent import run as langgraph_run
except ImportError:
    def langgraph_run(message, history=""): 
        return f"LangGraph stub response to: {message}"

# ================= CONFIG =================
UDP_IP = "127.0.0.1"  # Localhost for UDP stream
UDP_PORT = 25000      # Port used by Simulink exporter / sender
FEATURE_COUNT = 85    # Index + 84 features
SAMPLE_COUNT = 20     # Samples per batch
step_ms=1
SHAPE = (SAMPLE_COUNT, FEATURE_COUNT)  # Shared-memory array shape

FAULT_FOLDER = os.path.join(os.getcwd(), 'faults')
RECOMMENDATION_DIR = "ai_recommendations"

# ================= GLOBAL STATE =================
event_active = False
event_start_time = None
event_type = None
event_start_bus = 11  # Defaulting to Bus 11
event_start_sample_index = None

latest_data_chunk = {  # Last batch of data exposed to `/waveform_data`
    "x": [],
    "y": [],
    "labels": [],
    "current_end_index": 0,
}
data_lock = threading.Lock()

# ================= DB QUEUE (SINGLE WRITER) =================
# This prevents the crash by ensuring only one thread writes to the DB
db_queue = Queue()

def db_worker():
    """Background worker that serializes all DB writes through a single queue."""
    print("[DB Worker] Started. Waiting for events...")
    while True:
        item = db_queue.get()
        if item is None:
            break

        kind, fault = item
        try:
            if kind == "ongoing":
                vecdb.upsert_ongoing_event(fault)
            elif kind == "done":
                vecdb.upload_to_database_done(fault)
        except AttributeError as e:
            print(f"[DB ERROR] Function mismatch in database.py: {e}")
        except Exception as e:
            print(f"[DB ERROR] Update failed: {e}")
        finally:
            db_queue.task_done()

threading.Thread(target=db_worker, daemon=True).start()

# ================= LOGGER PROCESS =================
def logger_process(shmA_name, shmB_name, ready_A, ready_B):
    """Receive UDP frames and write them into the ping‑pong shared memory buffers."""
    try:
        shmA = SharedMemory(name=shmA_name)
        shmB = SharedMemory(name=shmB_name)
        bufA = np.ndarray(SHAPE, dtype=np.float64, buffer=shmA.buf)
        bufB = np.ndarray(SHAPE, dtype=np.float64, buffer=shmB.buf)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        print(f"[Logger] Listening on {UDP_IP}:{UDP_PORT}...")

        batch = 0
        idx = 0

        while True:
            buf = bufA if batch % 2 == 0 else bufB
            flag = ready_A if batch % 2 == 0 else ready_B

            for i in range(SAMPLE_COUNT):
                data, _ = sock.recvfrom(4096)
                values = struct.unpack("<84f", data)
                buf[i, :] = [idx] + list(values)
                idx += 1

            flag.set()
            batch += 1
    except Exception as e:
        print(f"[Logger] Crashed: {e}")

# ================= EVENT DETECTION =================
def detect_event(buf, sag_thr=0.9, swell_thr=1.1):
    """Simple threshold-based sag/swell detector on Phase A."""
    x = buf[:, 0]
    y = buf[:, 45]  # Monitoring Phase A (example index) bus 8 phase C

    if np.any(y < sag_thr):
        return x.tolist(), y.tolist(), "sag"
    if np.any(y > swell_thr):
        return x.tolist(), y.tolist(), "swell"

    return x.tolist(), y.tolist(), "normal"

def _write_event_file(filename, start_time, event_type, bus, status, duration=None):
    """Write event status to a JSON text file to be consumed by the frontend."""
    filepath = os.path.join(FAULT_FOLDER, filename)
    if not os.path.exists(FAULT_FOLDER): os.makedirs(FAULT_FOLDER)
    
    data = {
        "time": start_time,
        "type": "Voltage sag" if event_type == "sag" else "Voltage swell",
        "location": f"BUS {bus}",
        "status": status,
        "duration_ms": duration if duration is not None else "ongoing"
    }
    
    # Atomic write to avoid file corruption
    tmp = filepath + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, filepath)

# ================= BRIDGE THREAD =================
def bridge_thread(shmA_name, shmB_name, ready_A, ready_B):
    global event_active, event_start_time, event_type, event_start_sample_index

    print("[Bridge] Thread Started.")
    shmA = SharedMemory(name=shmA_name)
    shmB = SharedMemory(name=shmB_name)
    bufA = np.ndarray(SHAPE, dtype=np.float64, buffer=shmA.buf)
    bufB = np.ndarray(SHAPE, dtype=np.float64, buffer=shmB.buf)

    batch = 0

    while True:
        buf = bufA if batch % 2 == 0 else bufB
        flag = ready_A if batch % 2 == 0 else ready_B
        
        flag.wait()
        x, y, label = detect_event(buf)

        with data_lock:
            latest_data_chunk.update({
                "x": x,
                "y": y,
                "labels": [label] * len(x),
                "current_end_index": int(x[-1]),
            })

        # --- Event Logic ---
        if label in ("sag", "swell"):
            if not event_active:
                event_active = True
                event_start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                event_type = label
                event_start_sample_index = x[0]
                print(f"[FAULT] {label.upper()} started at {event_start_time}")
                
                # Write to filesystem for frontend
                _write_event_file(f"{event_start_time}.txt", event_start_time, label, event_start_bus, "ongoing")

            # Push ongoing update to DB Queue
            db_queue.put(("ongoing", {
                "time": event_start_time,
                "type": event_type,
                "location": f"BUS {event_start_bus}",
                "duration_ms": "ongoing",
            }))

        else:
            if event_active:
                duration = int(x[-1] - event_start_sample_index)
                print(f"[FAULT] {event_type.upper()} ended. Duration: {duration}")
                
                # Update filesystem file to 'final'
                _write_event_file(f"{event_start_time}.txt", event_start_time, event_type, event_start_bus, "final", duration)

                # Push completion to DB Queue
                db_queue.put(("done", {
                    "time": event_start_time,
                    "type": event_type,
                    "location": f"BUS {event_start_bus}",
                    "duration_ms": duration,
                }))
                event_active = False

        flag.clear()
        batch += 1

# ================= FLASK & ROUTES =================
app = Flask(__name__)
CORS(app)

def parse_md_file(filepath):
    """Helper to parse markdown recommendation files."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    content = re.sub(r"^'''md\s*|'''$", "", content, flags=re.MULTILINE).strip()
    time_match = re.search(r"\*\*Time:\*\*\s*(.+)", content)
    type_match = re.search(r"\*\*Type:\*\*\s*(.+)", content)
    return {
        "filename": os.path.basename(filepath),
        "time": time_match.group(1) if time_match else "Unknown",
        "type": type_match.group(1) if type_match else "Unknown"
    }

@app.route('/')
def home():
    """Serve main dashboard page."""
    return render_template('index.html')

@app.route('/waveform_data')
def waveform_data():
    with data_lock:
        return jsonify(latest_data_chunk)

@app.route('/reports')
def list_reports():
    if not os.path.exists("reports"): os.makedirs("reports")
    files = [f for f in os.listdir("reports") if f.endswith(".pdf")]
    return jsonify(files)

@app.route('/reports/<filename>')
def serve_report(filename):
    return send_from_directory("reports", filename)

@app.route('/faults')
def list_fault_files():
    if not os.path.exists(FAULT_FOLDER): os.makedirs(FAULT_FOLDER)
    txt_files = [f for f in os.listdir(FAULT_FOLDER) if f.endswith('.txt')]
    return jsonify(txt_files)

@app.route('/faults/<filename>')
def serve_fault_file(filename):
    return send_from_directory(FAULT_FOLDER, filename)

@app.route("/list_recommendations")
def list_recommendations():
    if not os.path.exists(RECOMMENDATION_DIR): os.makedirs(RECOMMENDATION_DIR)
    files_data = []
    for f in sorted(os.listdir(RECOMMENDATION_DIR), reverse=True):
        if f.endswith(".md"):
            filepath = os.path.join(RECOMMENDATION_DIR, f)
            parsed = parse_md_file(filepath)
            files_data.append(parsed)
    return jsonify(files_data)

@app.route('/recommendations/<path:filename>')
def get_recommendation(filename):
    filepath = os.path.join(RECOMMENDATION_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        content = re.sub(r"^'''md\s*|'''$", "", content, flags=re.MULTILINE).strip()
        return jsonify({"content": content})
    return jsonify({"error": "File not found"}), 404

@app.route('/generate_recommendation', methods=['POST'])
def generate_recommendation():
    try:
        subprocess.run([sys.executable, "langgraph_agent/recomendations.py"], check=True)
        return jsonify({"message": "New recommendation generated successfully!"})
    except subprocess.CalledProcessError as e:
        return jsonify({"message": f"Error: {e}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    # 1. Safe DB Update
    db_queue.put(("ongoing", {
        "time": event_start_time or "N/A",
        "type": event_type or "normal",
        "location": f"BUS {event_start_bus}",
        "duration_ms": "ongoing",
    }))

    # 2. Process Message
    data = request.json
    print(data)
    print(type(data))
    user_input = data.get('message', '')
    previous = data.get('history', '')

    plain_response = langgraph_run(user_input, previous)
    print(plain_response,"its a plain response")
    previous.append({"role": "user", "content": user_input})
    previous.append({"role": "assistant", "content": plain_response})

    updated_history = previous
    
    return jsonify({
        'response': markdown.markdown(plain_response),
        'history': updated_history
    })
# ================= COST MODEL (STATIC EXAMPLE ASSETS) =================


def compute_damage_probability(event_type, duration_ms, sensitivity):
    # Simple linear model (you can refine)
    sev = 0.6 if event_type == "sag" else 0.8
    dur_factor = min(duration_ms / 2000, 1.0)
    return round(sev * dur_factor * sensitivity, 3)

ASSET_DB = {
    "BUS 1": [
        {
            "name": "Induction Motor M1",
            "type": "motor",
            "rating_kw": 55,
            "asset_cost_rs": 450000,              # Replacement cost
            "process_cost_rs_per_min": 12000,     # Production loss per minute
            "sensitivity": 0.6
        },
        {
            "name": "Control Panel CP1",
            "type": "control_panel",
            "rating_kw": None,
            "asset_cost_rs": 150000,
            "process_cost_rs_per_min": 3000,
            "sensitivity": 0.35
        }
    ],

    "BUS 11": [
        {
            "name": "VFD Line-3",
            "type": "vfd",
            "rating_kw": 90,
            "asset_cost_rs": 850000,
            "process_cost_rs_per_min": 25000/60000,
            "sensitivity": 0.9
        },
        {
            "name": "Induction Motor M11",
            "type": "motor",
            "rating_kw": 75,
            "asset_cost_rs": 520000,
            "process_cost_rs_per_min": 15000/60000,
            "sensitivity": 0.7
        }
    ],

    "BUS 5": [
        {
            "name": "Packaging Robot R5",
            "type": "robot",
            "rating_kw": 30,
            "asset_cost_rs": 1200000,
            "process_cost_rs_per_min": 35000/(60000),       #its in per ms
            "sensitivity": 0.85
        }
    ],

    "BUS 7": [
        {
            "name": "CNC Machine CNC-7",
            "type": "cnc",
            "rating_kw": 45,
            "asset_cost_rs": 950000,
            "process_cost_rs_per_min": 28000/(60000),
            "sensitivity": 0.75
        }
    ]
}

# ================= DAMAGE COST ENDPOINT =================
@app.route('/extra_cost_data')
def extra_cost_data():
    """Reads last fault and returns asset-wise damage cost."""
    if not os.path.exists(FAULT_FOLDER):
        return jsonify({"assets": []})

    fault_files = [f for f in os.listdir(FAULT_FOLDER) if f.endswith(".txt")]
    if not fault_files:
        return jsonify({"assets": []})

    latest = sorted(fault_files)[-1]  # last detected fault

    with open(os.path.join(FAULT_FOLDER, latest), "r") as f:
        fault = json.load(f)

    bus = fault.get("location", "BUS 11")
    event_type = fault.get("type", "Voltage sag").replace("Voltage ", "")
    raw_duration = fault.get("duration_ms", 100)

    if raw_duration == "ongoing":
        duration = SAMPLE_COUNT*step_ms   # assume 2 sec exposure so far
    else:
        duration = int(float(raw_duration))

    assets = ASSET_DB.get(bus, [])
    output = []
    for a in assets:
        prob = compute_damage_probability(event_type, duration, a["sensitivity"])
        dmg_cost = prob * a["asset_cost_rs"]   # FIXED

        output.append({
            "name": a["name"],
            "sensitivity": a["sensitivity"],
            "replacement_cost": a["asset_cost_rs"],   # FIXED for UI column
            "damage_probability": prob,
            "damage_cost": round(dmg_cost, 2),
        })

    return jsonify({"assets": output})


# ================= MAIN =================
if __name__ == "__main__":
    # When run directly, start UDP logger process, bridge thread and Flask server.
    # Cleanup old shared memory
    try: SharedMemory(name="shmA").unlink()
    except: pass
    try: SharedMemory(name="shmB").unlink()
    except: pass

    shmA = SharedMemory(create=True, size=np.zeros(SHAPE).nbytes, name="shmA")
    shmB = SharedMemory(create=True, size=np.zeros(SHAPE).nbytes, name="shmB")

    ready_A = Event()
    ready_B = Event()
    p_logger = None

    try:
        p_logger = Process(target=logger_process, args=(shmA.name, shmB.name, ready_A, ready_B))
        p_logger.start()

        threading.Thread(target=bridge_thread, args=(shmA.name, shmB.name, ready_A, ready_B), daemon=True).start()

        print("Starting Flask Server...")
        app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if p_logger:
            p_logger.terminate()
            p_logger.join()
        shmA.close()
        shmA.unlink()
        shmB.close()
        shmB.unlink()