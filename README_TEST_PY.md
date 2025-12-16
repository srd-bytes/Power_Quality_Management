**Overview**
- **What:** Run the `test.py` Flask app that listens for UDP waveform data, detects events (sag/swell), writes fault files, and exposes a small API/UI.
- **Main script:** `test.py`

**Prerequisites**
- **Python:** 3.10+ recommended.
- **Install dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Quick Start**
- Start the service (runs logger + bridge + Flask server):

```bash
python test.py
```

- The Flask server will start on `http://0.0.0.0:5000`.

**What the service expects**
- A logger process binds to UDP `127.0.0.1:25000` and expects datagrams containing 84 floats packed with `struct.pack('<84f', *values)` (see `test.py` where `struct.unpack("<84f", data)` is used).
- `test.py` creates two POSIX shared memory blocks named `shmA` and `shmB` to shuttle data between the logger and the bridge thread. The script will attempt to unlink old shared memory names on startup.

**Testing locally (UDP sender example)**
If you don't have a real data source, use this small Python snippet to send a single sample (84 floats) to the logger loop:

```python
import socket
import struct

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ('127.0.0.1', 25000)

# Prepare 84 floats. First float is an index (e.g. 0.0), remaining 83 are measurements.
vals = [0.0] + [1.0] * 83
data = struct.pack('<84f', *vals)
sock.sendto(data, addr)
```

Send this repeatedly (in a loop) to exercise the logger/bridge and trigger event detection.

**API Endpoints (useful ones)**
- `GET /waveform_data` — current buffered waveform chunk (JSON).
- `GET /faults` — list fault `.txt` files produced.
- `GET /recommendations/<file>` — read a recommendation `.md` file from `ai_recommendations`.
- `POST /generate_recommendation` — runs the `langgraph_agent/recomendations.py` script.
- `POST /chat` — pass JSON `{"message": "...", "history": "..."}` to get agent response.

**Stopping and Cleanup**
- Use `Ctrl+C` to stop — the script handles KeyboardInterrupt and will terminate the logger process and unlink shared memory blocks.

**Troubleshooting**
- Port in use: if port `5000` or UDP `25000` is already used, stop the conflicting service or change the port in `test.py`.
- Dependency errors: ensure you installed from `requirements.txt` inside the virtualenv.
- Shared memory errors: older `shmA/shmB` name collisions may cause startup issues; `test.py` already attempts to unlink them. If issues persist, reboot or inspect `/dev/shm`.

**Notes**
- The code expects 84 floats per UDP packet (`struct.unpack('<84f', data)`). Modify `FEATURE_COUNT` / `SAMPLE_COUNT` in `test.py` only if you know the source format.
- `test.py` also writes to `faults/` and `reports/` directories; ensure the process has write permissions.

**Next steps**
- If you want, I can add a small `send_test_packets.py` helper file and a simple systemd unit / supervisor example to run `test.py` as a service.