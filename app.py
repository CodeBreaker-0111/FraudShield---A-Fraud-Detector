"""
app.py
------
Flask backend for the Card Fraud Detection System.

Run with:
    python app.py
Then open: http://127.0.0.1:5000
"""

import json
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODELS_DIR = Path("models")

# ─── Load model at startup ────────────────────────────────────────────────────

model   = None
scaler  = None
features = None
metrics  = None

def load_model():
    global model, scaler, features, metrics
    model_path   = MODELS_DIR / "fraud_model.pkl"
    scaler_path  = MODELS_DIR / "scaler.pkl"
    features_path= MODELS_DIR / "features.json"
    metrics_path = MODELS_DIR / "metrics.json"

    if not model_path.exists():
        print("WARNING: Model not found. Run train_model.py first.")
        return False

    model   = joblib.load(model_path)
    scaler  = joblib.load(scaler_path)
    with open(features_path) as f:
        features = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    print("Model loaded successfully.")
    return True


# ─── Merchant risk mapping ────────────────────────────────────────────────────

MERCHANT_RISK_MAP = {
    "grocery":     0.05,
    "restaurant":  0.07,
    "pharmacy":    0.08,
    "fuel":        0.10,
    "travel":      0.20,
    "online":      0.25,
    "electronics": 0.30,
    "atm":         0.35,
    "jewelry":     0.45,
    "gaming":      0.55,
}


# ─── Feature builder ─────────────────────────────────────────────────────────

def build_features(data):
    """
    Converts raw form input into the feature vector the model expects.
    Works for both synthetic-trained and kaggle-trained models.
    """
    amount          = float(data.get("amount", 0))
    hour            = int(data.get("hour", 12))
    distance        = float(data.get("distance_from_home", 0))
    txn_per_hour    = int(data.get("txn_per_hour", 1))
    avg_spend       = float(data.get("avg_monthly_spend", 15000))
    new_device      = int(data.get("new_device", 0))
    merchant_cat    = data.get("merchant_category", "online")
    card_type       = 0 if data.get("card_type", "credit") == "credit" else 1
    international   = int(data.get("international", 0))

    merchant_risk   = MERCHANT_RISK_MAP.get(merchant_cat, 0.15)
    amount_ratio    = amount / max(avg_spend / 30, 1)
    is_night        = 1 if 1 <= hour <= 5 else 0
    high_velocity   = 1 if txn_per_hour >= 4 else 0
    far_from_home   = 1 if distance > 100 else 0
    risk_amount_combo = merchant_risk * amount_ratio

    # Map feature names to values
    feat_map = {
        "amount":              amount,
        "hour":                hour,
        "distance_from_home":  distance,
        "txn_per_hour":        txn_per_hour,
        "avg_monthly_spend":   avg_spend,
        "new_device":          new_device,
        "merchant_risk":       merchant_risk,
        "card_type":           card_type,
        "international":       international,
        "amount_ratio":        amount_ratio,
        "is_night":            is_night,
        "high_velocity":       high_velocity,
        "far_from_home":       far_from_home,
        "risk_amount_combo":   risk_amount_combo,
        # Kaggle features — filled with 0 if not applicable
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "amount_norm":  (amount - 88) / 250,
        "time_norm":    0.0,
        "amount_log":   np.log1p(amount),
    }

    vector = [feat_map.get(f, 0.0) for f in features]
    return vector, {
        "amount":        amount,
        "hour":          hour,
        "distance":      distance,
        "txn_per_hour":  txn_per_hour,
        "merchant_risk": merchant_risk,
        "amount_ratio":  round(amount_ratio, 2),
        "is_night":      is_night,
        "new_device":    new_device,
        "international": international,
        "card_type":     data.get("card_type", "credit"),
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Run: python train_model.py first, then restart app.py"
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    vector, parsed = build_features(data)

    X = np.array(vector).reshape(1, -1)
    X_scaled = scaler.transform(X)

    fraud_proba = float(model.predict_proba(X_scaled)[0][1])
    prediction  = int(fraud_proba >= 0.5)

    # Risk level
    if fraud_proba < 0.30:
        risk_level = "low"
        verdict    = "Transaction looks legitimate. No action needed."
    elif fraud_proba < 0.65:
        risk_level = "medium"
        verdict    = "Suspicious patterns detected. Send OTP verification to cardholder."
    else:
        risk_level = "high"
        verdict    = "High fraud probability! Block this transaction and alert the cardholder immediately."

    # Explain which factors contributed most
    risk_factors = _explain(parsed, fraud_proba)

    return jsonify({
        "fraud_probability": round(fraud_proba * 100, 1),
        "prediction":        prediction,
        "risk_level":        risk_level,
        "verdict":           verdict,
        "risk_factors":      risk_factors,
        "parsed_features":   parsed,
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if metrics is None:
        return jsonify({"error": "Model not trained yet."}), 503
    return jsonify(metrics)


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    """Predict on multiple transactions at once (for the dashboard demo)."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    transactions = request.get_json()
    if not isinstance(transactions, list):
        return jsonify({"error": "Expected a list of transactions"}), 400

    results = []
    for txn in transactions:
        vector, parsed = build_features(txn)
        X = np.array(vector).reshape(1, -1)
        X_scaled = scaler.transform(X)
        proba = float(model.predict_proba(X_scaled)[0][1])

        results.append({
            "id":               txn.get("id", "—"),
            "fraud_probability": round(proba * 100, 1),
            "risk_level":        "high" if proba >= 0.65 else "medium" if proba >= 0.30 else "low",
            "amount":            txn.get("amount"),
            "card_type":         txn.get("card_type", "credit"),
        })

    return jsonify(results)


# ─── Explainability helper ────────────────────────────────────────────────────

def _explain(parsed, proba):
    factors = []

    if parsed["amount_ratio"] > 1.5:
        factors.append({"label": "High spend ratio", "value": f"{parsed['amount_ratio']}x avg daily", "level": "high"})
    elif parsed["amount_ratio"] > 0.8:
        factors.append({"label": "Moderate spend ratio", "value": f"{parsed['amount_ratio']}x avg daily", "level": "medium"})
    else:
        factors.append({"label": "Normal amount", "value": f"{parsed['amount_ratio']}x avg daily", "level": "low"})

    if parsed["is_night"]:
        factors.append({"label": "Night transaction (1–5 AM)", "value": f"{parsed['hour']}:00", "level": "high"})
    else:
        factors.append({"label": "Transaction hour", "value": f"{parsed['hour']}:00", "level": "low"})

    if parsed["distance"] > 100:
        factors.append({"label": "Far from home", "value": f"{parsed['distance']} km", "level": "high"})
    elif parsed["distance"] > 30:
        factors.append({"label": "Moderate distance", "value": f"{parsed['distance']} km", "level": "medium"})
    else:
        factors.append({"label": "Near home", "value": f"{parsed['distance']} km", "level": "low"})

    if parsed["txn_per_hour"] >= 4:
        factors.append({"label": "High velocity", "value": f"{parsed['txn_per_hour']} txns/hr", "level": "high"})
    elif parsed["txn_per_hour"] >= 2:
        factors.append({"label": "Some velocity", "value": f"{parsed['txn_per_hour']} txns/hr", "level": "medium"})
    else:
        factors.append({"label": "Normal velocity", "value": f"{parsed['txn_per_hour']} txns/hr", "level": "low"})

    m_risk = parsed["merchant_risk"]
    m_level = "high" if m_risk >= 0.35 else "medium" if m_risk >= 0.15 else "low"
    factors.append({"label": "Merchant risk", "value": f"{int(m_risk*100)}%", "level": m_level})

    if parsed["new_device"]:
        factors.append({"label": "New device/location", "value": "Yes", "level": "high"})
    else:
        factors.append({"label": "Known device", "value": "No", "level": "low"})

    if parsed["international"]:
        factors.append({"label": "International transaction", "value": "Yes", "level": "high" if parsed["card_type"] == "debit" else "medium"})

    return factors


# ─── Start ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    print("\nStarting server at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)