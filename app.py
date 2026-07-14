"""
app.py – Flask backend for Credit Card Approval / Fraud Prediction
Loads the best trained model and exposes a prediction endpoint.
"""

import os
import pickle

import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# Load model once at startup
try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    print(f"[INFO] Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    MODEL = None
    print("[WARNING] Model file not found. Run credit_card_approval.py first.")


# ---------------------------------------------------------------------------
# Feature list (matches training order after removing Amount & Time columns
# and using Amount_Scaled / Time_Scaled instead)
# V1–V28 + Amount_Scaled + Time_Scaled  → 30 features
# ---------------------------------------------------------------------------
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount_Scaled", "Time_Scaled"]

FEATURE_LABELS = {
    "V1":  "V1 (PCA Feature 1)",
    "V2":  "V2 (PCA Feature 2)",
    "V3":  "V3 (PCA Feature 3)",
    "V4":  "V4 (PCA Feature 4)",
    "V5":  "V5 (PCA Feature 5)",
    "V6":  "V6 (PCA Feature 6)",
    "V7":  "V7 (PCA Feature 7)",
    "V8":  "V8 (PCA Feature 8)",
    "V9":  "V9 (PCA Feature 9)",
    "V10": "V10 (PCA Feature 10)",
    "V11": "V11 (PCA Feature 11)",
    "V12": "V12 (PCA Feature 12)",
    "V13": "V13 (PCA Feature 13)",
    "V14": "V14 (PCA Feature 14)",
    "V15": "V15 (PCA Feature 15)",
    "V16": "V16 (PCA Feature 16)",
    "V17": "V17 (PCA Feature 17)",
    "V18": "V18 (PCA Feature 18)",
    "V19": "V19 (PCA Feature 19)",
    "V20": "V20 (PCA Feature 20)",
    "V21": "V21 (PCA Feature 21)",
    "V22": "V22 (PCA Feature 22)",
    "V23": "V23 (PCA Feature 23)",
    "V24": "V24 (PCA Feature 24)",
    "V25": "V25 (PCA Feature 25)",
    "V26": "V26 (PCA Feature 26)",
    "V27": "V27 (PCA Feature 27)",
    "V28": "V28 (PCA Feature 28)",
    "Amount_Scaled": "Transaction Amount (Scaled)",
    "Time_Scaled":   "Transaction Time (Scaled)",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """Landing page with project overview."""
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
def predict_form():
    """Render the prediction input form."""
    return render_template("index.html",
                           features=FEATURE_NAMES,
                           labels=FEATURE_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    """Collect form values, run model, redirect to result page."""
    if MODEL is None:
        return render_template("result.html",
                               result="ERROR",
                               confidence=0,
                               message="Model not loaded. Please train the model first.",
                               css_class="error")

    try:
        values = [float(request.form.get(feat, 0)) for feat in FEATURE_NAMES]
        input_array = np.array(values).reshape(1, -1)

        prediction  = MODEL.predict(input_array)[0]
        probability = MODEL.predict_proba(input_array)[0]

        is_fraud    = bool(prediction == 1)
        confidence  = round(float(probability[1] if is_fraud else probability[0]) * 100, 2)

        if is_fraud:
            result    = "REJECTED – Fraudulent Transaction Detected"
            message   = (f"This transaction has been flagged as potentially fraudulent "
                         f"with {confidence}% confidence. The application is rejected.")
            css_class = "rejected"
        else:
            result    = "APPROVED – Legitimate Transaction"
            message   = (f"This transaction appears legitimate with {confidence}% confidence. "
                         f"The credit card application is approved.")
            css_class = "approved"

        return render_template("result.html",
                               result=result,
                               confidence=confidence,
                               message=message,
                               css_class=css_class)

    except Exception as exc:
        return render_template("result.html",
                               result="ERROR",
                               confidence=0,
                               message=f"Prediction error: {str(exc)}",
                               css_class="error")


@app.route("/about")
def about():
    """Project information page."""
    return render_template("home.html", section="about")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
