import os
import time
import threading
import csv
from datetime import datetime
import joblib
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, Response
from arduino.app_utils import Bridge

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saltbae_rf_model.joblib")
rf_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Demo scale: how many "grams" correspond to one second of open gate.
# Higher number = shorter open times for the same prediction.
FLOW_RATE_G_PER_SEC = 15.0
MIN_OPEN_MS = 400    # shortest open time, so small doses still visibly move the servo
MAX_OPEN_MS = 5000   # longest open time (keep equal to MAX_OPEN_MS in the sketch)

# ---- Training-data log -------------------------------------------------
# Saved in the app folder (not inside python/), so it survives restarts.
# On the board: /home/arduino/ArduinoApps/saltbae/data/samples.csv
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "samples.csv")
DATA_COLUMNS = ["timestamp", "ice_thickness_mm", "pavement_temp_c",
                "humidity_pct", "required_salt_grams", "raw_distance_mm"]
data_lock = threading.Lock()

def append_sample(grams):
    row = [
        datetime.now().isoformat(timespec="seconds"),
        round(state["ice_thickness_mm"], 2),
        state["temperature_c"],
        state["humidity_pct"],
        round(grams, 2),
        round(state["distance_mm"], 1),
    ]
    with data_lock:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        new_file = not os.path.exists(DATA_PATH)
        with open(DATA_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(DATA_COLUMNS)
            w.writerow(row)
        with open(DATA_PATH) as f:
            return sum(1 for _ in f) - 1   # rows saved so far

state = {
    "ice_thickness_mm": 0.0,
    "distance_mm": -1.0,
    "temperature_c": -3.0,
    "humidity_pct": 80.0,
    "predicted_grams": 0.0,
    "gate_duration_ms": 0,
    "last_dispense_time": 0.0
}

def predict_salt(ice_mm, temp_c, hum_pct):
    if ice_mm < 1.0 or temp_c > 1.0:
        return 0.0
    if rf_model is not None:
        features = pd.DataFrame(
            [[ice_mm, temp_c, hum_pct]],
            columns=["ice_thickness_mm", "pavement_temp_c", "humidity_pct"]
        )
        return float(rf_model.predict(features)[0])
    return float((ice_mm * 1.4) + (ice_mm * abs(temp_c) * 0.45) + (hum_pct / 100.0 * 1.2))

def autonomous_loop():
    while True:
        try:
            ice_mm = float(Bridge.call("get_ice_thickness"))  # type: ignore
            # raw distance from the SAME measurement (-1 means no echo)
            dist_mm = float(Bridge.call("get_last_distance_mm"))  # type: ignore
            state["ice_thickness_mm"] = ice_mm
            state["distance_mm"] = dist_mm
            print("distance_mm =", round(dist_mm, 1), "| ice_mm =", round(ice_mm, 1), flush=True)

            temp = state["temperature_c"]
            hum = state["humidity_pct"]

            grams = predict_salt(ice_mm, temp, hum)
            state["predicted_grams"] = max(0.0, grams)
            duration_sec = state["predicted_grams"] / FLOW_RATE_G_PER_SEC
            ms = int(duration_sec * 1000)
            if ms > 0:
                ms = max(MIN_OPEN_MS, min(ms, MAX_OPEN_MS))
            state["gate_duration_ms"] = ms

            now = time.time()
            if ice_mm >= 3.0 and (now - state["last_dispense_time"] > 6.0) and state["gate_duration_ms"] > 0:
                print("dispensing for", state["gate_duration_ms"], "ms", flush=True)
                Bridge.call("dispense_salt", state["gate_duration_ms"])
                state["last_dispense_time"] = time.time()
        except Exception as e:
            print("autonomous_loop error:", repr(e), flush=True)
        time.sleep(0.4)

threading.Thread(target=autonomous_loop, daemon=True).start()

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>SaltBae Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; max-width: 480px; margin: auto; }
        .card { background: #1e293b; border-radius: 12px; padding: 18px; margin-bottom: 14px; }
        h1 { font-size: 1.4rem; color: #38bdf8; text-align: center; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
        .stat-box { background: #0f172a; padding: 12px; border-radius: 8px; text-align: center; }
        .stat-val { font-size: 1.6rem; font-weight: bold; color: #38bdf8; }
        .stat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }
        .slider-group { margin-bottom: 16px; }
        label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 6px; }
        input[type=range] { width: 100%; accent-color: #38bdf8; }
        button { width: 100%; background: #0284c7; color: white; border: none; padding: 12px; border-radius: 8px; font-size: 1rem; font-weight: bold; cursor: pointer; }
    </style>
</head>
<body>
    <h1>SaltBae</h1>
    <div class="card">
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-val" id="ice-val">-- mm</div>
                <div class="stat-label">HC-SR04 Ice</div>
            </div>
            <div class="stat-box">
                <div class="stat-val" id="pred-val">-- g</div>
                <div class="stat-label">Predicted Salt</div>
            </div>
        </div>
        <div class="stat-box">
            <div class="stat-val" id="gate-val" style="color: #4ade80;">-- ms</div>
            <div class="stat-label">Gate Pulse Duration</div>
        </div>
        <div class="stat-box" style="margin-top: 10px;">
            <div class="stat-val" id="dist-val">-- mm</div>
            <div class="stat-label">Raw Sensor Distance</div>
        </div>
    </div>
    <div class="card">
        <div class="slider-group">
            <label><span>Pavement Temp:</span> <strong id="temp-label">-3.0 °C</strong></label>
            <input type="range" id="temp-slider" min="-15" max="5" step="0.5" value="-3.0">
        </div>
        <div class="slider-group">
            <label><span>Humidity:</span> <strong id="hum-label">80 %</strong></label>
            <input type="range" id="hum-slider" min="40" max="100" step="1" value="80">
        </div>
        <button onclick="triggerDispense()">Dispense Calculated Dose</button>
    </div>
    <div class="card">
        <label><span>Salt that actually cleared the ice (g)</span></label>
        <input type="number" id="grams-input" min="0" max="500" step="0.1" placeholder="e.g. 12.5"
               style="width:100%; box-sizing:border-box; padding:10px; border-radius:8px; border:none; margin-bottom:10px; font-size:1rem;">
        <button onclick="saveSample()">Save Training Sample</button>
        <div id="sample-status" style="font-size:0.8rem; color:#94a3b8; margin-top:8px; text-align:center;">
            Saves the current ice reading, temperature and humidity with the grams you enter.
            <a href="/api/samples.csv" style="color:#38bdf8;">Download CSV</a>
        </div>
    </div>
    <script>
        const tempSlider = document.getElementById('temp-slider');
        const humSlider = document.getElementById('hum-slider');
        function updateSliders() {
            document.getElementById('temp-label').innerText = `${tempSlider.value} °C`;
            document.getElementById('hum-label').innerText = `${humSlider.value} %`;
            fetch('/api/update_env', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ temperature: parseFloat(tempSlider.value), humidity: parseFloat(humSlider.value) })
            });
        }
        tempSlider.oninput = updateSliders;
        humSlider.oninput = updateSliders;
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                document.getElementById('ice-val').innerText = `${data.ice_thickness.toFixed(1)} mm`;
                document.getElementById('pred-val').innerText = `${data.predicted_grams.toFixed(2)} g`;
                document.getElementById('gate-val').innerText = `${data.gate_duration_ms} ms`;
                document.getElementById('dist-val').innerText = data.distance_mm < 0 ? 'no echo' : `${data.distance_mm.toFixed(1)} mm`;
            } catch (err) {}
        }
        setInterval(fetchTelemetry, 400);
        function triggerDispense() { fetch('/api/dispense', { method: 'POST' }); }
        async function saveSample() {
            const status = document.getElementById('sample-status');
            const grams = parseFloat(document.getElementById('grams-input').value);
            if (isNaN(grams)) { status.innerText = 'Enter the grams first.'; return; }
            try {
                const res = await fetch('/api/log_sample', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ grams: grams })
                });
                const data = await res.json();
                status.innerText = res.ok
                    ? `Saved. ${data.rows} sample(s) so far (ice ${data.ice_mm} mm, ${data.temp_c} C, ${data.hum_pct}%).`
                    : `Not saved: ${data.error}`;
            } catch (err) { status.innerText = 'Not saved: could not reach the app.'; }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/update_env", methods=["POST"])
def update_env():
    data = request.get_json()
    state["temperature_c"] = float(data.get("temperature", state["temperature_c"]))
    state["humidity_pct"] = float(data.get("humidity", state["humidity_pct"]))
    return jsonify({"status": "ok"})

@app.route("/api/telemetry")
def telemetry():
    return jsonify({
        "ice_thickness": state["ice_thickness_mm"],
        "distance_mm": state["distance_mm"],
        "predicted_grams": state["predicted_grams"],
        "gate_duration_ms": state["gate_duration_ms"]
    })

@app.route("/api/dispense", methods=["POST"])
def dispense():
    ms = state["gate_duration_ms"] if state["gate_duration_ms"] > 0 else 1000
    Bridge.call("dispense_salt", ms)
    return jsonify({"status": "dispensed", "duration_ms": ms})

@app.route("/api/log_sample", methods=["POST"])
def log_sample():
    data = request.get_json(silent=True) or {}
    try:
        grams = float(data.get("grams"))
    except (TypeError, ValueError):
        return jsonify({"error": "grams must be a number"}), 400
    if not (0.0 <= grams <= 500.0):
        return jsonify({"error": "grams must be between 0 and 500"}), 400
    if state["ice_thickness_mm"] <= 0.0:
        return jsonify({"error": "no ice reading yet (sensor reads 0 mm)"}), 400
    try:
        rows = append_sample(grams)
    except OSError as e:
        return jsonify({"error": f"could not write file: {e}"}), 500
    return jsonify({"status": "saved", "rows": rows,
                    "ice_mm": round(state["ice_thickness_mm"], 1),
                    "temp_c": state["temperature_c"],
                    "hum_pct": state["humidity_pct"]})

@app.route("/api/samples.csv")
def samples_csv():
    with data_lock:
        if not os.path.exists(DATA_PATH):
            return Response("No samples saved yet.\n", mimetype="text/plain")
        with open(DATA_PATH) as f:
            body = f.read()
    return Response(body, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=samples.csv"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
