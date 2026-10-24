import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML + CSS Layout embedded directly inside app.py
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Material Transformation Predictor</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 32px;
            max-width: 550px;
            width: 100%;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        }

        h1 {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 8px;
            text-align: center;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-align: center;
            margin-bottom: 28px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-main);
        }

        input[type="number"] {
            width: 100%;
            padding: 12px 16px;
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input[type="number"]:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }

        button {
            width: 100%;
            padding: 14px;
            background-color: var(--accent);
            color: #0f172a;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            margin-top: 10px;
        }

        button:hover {
            background-color: var(--accent-hover);
            color: #ffffff;
        }

        button:active {
            transform: scale(0.98);
        }

        .result-box {
            margin-top: 28px;
            padding: 16px;
            border-radius: 8px;
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            text-align: center;
            display: {% if prediction is not none %} block {% else %} none {% endif %};
        }

        .result-box h2 {
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .result-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--accent);
        }

        .error-msg {
            color: #f87171;
            font-size: 0.875rem;
            text-align: center;
            margin-top: 12px;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Material Predictor</h1>
    <p class="subtitle">Enter material parameters to calculate output classification</p>

    <form method="POST" action="/predict">
        <div class="form-group">
            <label for="temp">Temperature (°C)</label>
            <input type="number" step="any" id="temp" name="temp" value="{{ temp }}" required>
        </div>

        <div class="form-group">
            <label for="pressure">Pressure (kPa)</label>
            <input type="number" step="any" id="pressure" name="pressure" value="{{ pressure }}" required>
        </div>

        <div class="form-group">
            <label for="temp_x_pressure">Temperature × Pressure</label>
            <input type="number" step="any" id="temp_x_pressure" name="temp_x_pressure" value="{{ temp_x_pressure }}" placeholder="Auto-calculates if empty">
        </div>

        <div class="form-group">
            <label for="fusion">Material Fusion Metric</label>
            <input type="number" step="any" id="fusion" name="fusion" value="{{ fusion }}" required>
        </div>

        <div class="form-group">
            <label for="transformation">Material Transformation Metric</label>
            <input type="number" step="any" id="transformation" name="transformation" value="{{ transformation }}" required>
        </div>

        <button type="submit">Run Prediction</button>
    </form>

    {% if error %}
        <div class="error-msg">{{ error }}</div>
    {% endif %}

    <div class="result-box">
        <h2>Prediction Result</h2>
        <div class="result-value">Class: {{ prediction }}</div>
    </div>
</div>

<script>
    // Auto-calculate Temperature x Pressure if both inputs change
    const tempInput = document.getElementById('temp');
    const pressureInput = document.getElementById('pressure');
    const tempXPressureInput = document.getElementById('temp_x_pressure');

    function updateInteraction() {
        const t = parseFloat(tempInput.value);
        const p = parseFloat(pressureInput.value);
        if (!isNaN(t) && !isNaN(p)) {
            tempXPressureInput.value = (t * p).toFixed(4);
        }
    }

    tempInput.addEventListener('input', updateInteraction);
    pressureInput.addEventListener('input', updateInteraction);
</script>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_LAYOUT, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_LAYOUT, 
            prediction=None, 
            error="Model file (model.pkl) not found on server."
        )

    try:
        temp = float(request.form.get("temp", 0))
        pressure = float(request.form.get("pressure", 0))
        
        # Take input or compute interaction term automatically
        txp_val = request.form.get("temp_x_pressure")
        if txp_val and txp_val.strip() != "":
            temp_x_pressure = float(txp_val)
        else:
            temp_x_pressure = temp * pressure

        fusion = float(request.form.get("fusion", 0))
        transformation = float(request.form.get("transformation", 0))

        # Format input array for Scikit-Learn (1 sample, 5 features)
        features = np.array([[temp, pressure, temp_x_pressure, fusion, transformation]])
        
        # Generate prediction
        pred = model.predict(features)[0]

        return render_template_string(
            HTML_LAYOUT,
            prediction=pred,
            temp=temp,
            pressure=pressure,
            temp_x_pressure=temp_x_pressure,
            fusion=fusion,
            transformation=transformation
        )

    except Exception as e:
        return render_template_string(
            HTML_LAYOUT, 
            prediction=None, 
            error=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
