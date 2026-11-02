"""
FraudShield AI - Machine Learning Prediction Module
===================================================
College CIA-2 Project: AI-Powered Financial Fraud Detection System
Algorithm: Scikit-learn Isolation Forest (Inference Engine)

This module:
1. Loads the serialized Isolation Forest model and preprocessor metadata.
2. Validates incoming transaction parameters.
3. Computes the mathematical anomaly decision score.
4. Generates a calibrated 0 to 100 Risk Score.
5. Assigns a categorical risk tier: Low, Medium, or High.
6. Generates human-understandable explanations for any flagged behavior.
7. Emphasizes that anomaly detection flags statistical deviations, not legal proof of fraud.
"""

import os
import numpy as np
import pandas as pd
import joblib

# Paths to saved model artifacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "ml_model", "saved_models")
MODEL_FILE = os.path.join(SAVED_MODELS_DIR, "isolation_forest.joblib")
PREPROCESSOR_FILE = os.path.join(SAVED_MODELS_DIR, "preprocessor.joblib")

# Cached model & preprocessor in memory for fast repeated inference
_MODEL_CACHE = None
_PREPROCESSOR_CACHE = None


def load_model_artifacts():
    """
    Load the saved Isolation Forest model and preprocessor metadata.
    Uses an in-memory cache to avoid reading from disk on every prediction.
    """
    global _MODEL_CACHE, _PREPROCESSOR_CACHE

    if _MODEL_CACHE is not None and _PREPROCESSOR_CACHE is not None:
        return _MODEL_CACHE, _PREPROCESSOR_CACHE

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_FILE}'. "
            "Please train the model first by running: python ml_model/train.py"
        )

    if not os.path.exists(PREPROCESSOR_FILE):
        raise FileNotFoundError(
            f"Preprocessor metadata not found at '{PREPROCESSOR_FILE}'. "
            "Please train the model first by running: python ml_model/train.py"
        )

    _MODEL_CACHE = joblib.load(MODEL_FILE)
    _PREPROCESSOR_CACHE = joblib.load(PREPROCESSOR_FILE)

    return _MODEL_CACHE, _PREPROCESSOR_CACHE


def validate_transaction_input(data):
    """
    Validate and sanitize input transaction attributes.
    Returns cleaned dictionary or raises ValueError.
    """
    if not isinstance(data, dict):
        raise ValueError("Transaction input must be a dictionary.")

    # 1. Amount validation
    if "amount" not in data:
        raise ValueError("Missing required field: 'amount'")
    try:
        amount = float(data["amount"])
    except (TypeError, ValueError):
        raise ValueError(f"Invalid amount value: '{data.get('amount')}'. Must be a valid positive number.")
    if amount <= 0:
        raise ValueError("Transaction amount must be strictly greater than 0.")

    # 2. Transaction hour validation (0 to 23)
    hour = data.get("transaction_hour", 12)
    try:
        hour = int(hour)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid transaction_hour value: '{hour}'. Must be an integer between 0 and 23.")
    if not (0 <= hour <= 23):
        raise ValueError(f"transaction_hour must be between 0 and 23. Received: {hour}")

    # 3. Transaction frequency validation (>= 1)
    frequency = data.get("transaction_frequency", 1)
    try:
        frequency = int(frequency)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid transaction_frequency value: '{frequency}'. Must be an integer >= 1.")
    if frequency < 1:
        raise ValueError(f"transaction_frequency must be at least 1. Received: {frequency}")

    # 4. Unusual location indicator validation (0 or 1)
    location = data.get("is_unusual_location", 0)
    try:
        location = 1 if int(location) == 1 else 0
    except (TypeError, ValueError):
        raise ValueError(f"Invalid is_unusual_location value: '{location}'. Must be 0 or 1.")

    return {
        "amount": amount,
        "transaction_hour": hour,
        "transaction_frequency": frequency,
        "is_unusual_location": location
    }


def explain_flagged_factors(cleaned_data, risk_level):
    """
    Generate plain-language explanations detailing which factors contributed
    to an anomalous or elevated risk score.
    """
    reasons = []

    # Check Amount Factor
    if cleaned_data["amount"] >= 10000.0:
        reasons.append(
            f"High Transaction Amount: ${cleaned_data['amount']:,.2f} is exceptionally large compared to typical cardholder activity."
        )
    elif cleaned_data["amount"] >= 2500.0:
        reasons.append(
            f"Elevated Transaction Amount: ${cleaned_data['amount']:,.2f} exceeds standard baseline purchases."
        )

    # Check Hour Factor (Night-time anomalies)
    if cleaned_data["transaction_hour"] in [0, 1, 2, 3, 4, 5]:
        reasons.append(
            f"Unusual Transaction Time: Initiated at {cleaned_data['transaction_hour']:02d}:00 hours (overnight/off-peak)."
        )

    # Check Frequency Factor (Velocity attacks / card testing)
    if cleaned_data["transaction_frequency"] >= 15:
        reasons.append(
            f"Rapid Velocity Spike: {cleaned_data['transaction_frequency']} transactions performed within 24 hours."
        )
    elif cleaned_data["transaction_frequency"] >= 8:
        reasons.append(
            f"High Transaction Frequency: {cleaned_data['transaction_frequency']} transactions in 24 hours."
        )

    # Check Location Factor
    if cleaned_data["is_unusual_location"] == 1:
        reasons.append(
            "Geographic/Device Mismatch: Transaction originated from an unrecognized foreign location or untrusted device."
        )

    # If no individual extreme triggered but combination created an anomaly
    if not reasons and risk_level != "Low":
        reasons.append(
            "Multivariate Outlier: The combination of amount, timing, and velocity forms an uncommon statistical pattern."
        )
    elif not reasons and risk_level == "Low":
        reasons.append(
            "Normal Behavioral Profile: Amount, time, frequency, and location align with typical legitimate banking activity."
        )

    return reasons


def predict_transaction(transaction_data):
    """
    Evaluate a transaction using the trained Isolation Forest model.

    Parameters:
        transaction_data (dict): Dictionary with keys:
            - 'amount' (float): Transaction amount.
            - 'transaction_hour' (int, optional): 0-23 (default 12).
            - 'transaction_frequency' (int, optional): >= 1 (default 1).
            - 'is_unusual_location' (int, optional): 0 or 1 (default 0).

    Returns:
        dict:
            - is_anomaly (bool): True if flagged as an anomaly, False otherwise.
            - prediction (str): 'Suspected Fraud' or 'Normal'.
            - risk_score (float): Calibrated score from 0.0 to 100.0.
            - risk_level (str): 'Low', 'Medium', or 'High'.
            - anomaly_decision_score (float): Raw Isolation Forest decision function value.
            - explanations (list): Explanatory reasons for the decision.
            - input_data (dict): Cleaned input features.
            - disclaimer (str): Clarification on anomaly detection vs. legal proof.
    """
    # 1. Load artifacts (with caching)
    model, preprocessor = load_model_artifacts()

    # 2. Validate input
    cleaned_input = validate_transaction_input(transaction_data)

    # 3. Create DataFrame with proper column ordering
    feature_cols = preprocessor["features"]
    df_input = pd.DataFrame([cleaned_input])[feature_cols]

    # 4. Standardize input features using the saved scaler
    scaler = preprocessor["scaler"]
    X_scaled = scaler.transform(df_input)

    # 5. Model Inference
    # predict() returns 1 for normal (inlier), -1 for anomaly (outlier)
    raw_pred = model.predict(X_scaled)[0]
    is_anomaly = bool(raw_pred == -1)

    # decision_function() returns the anomaly score
    # Lower / negative scores mean higher anomaly risk
    raw_score = float(model.decision_function(X_scaled)[0])

    # 6. Calibrated Risk Score (0.0 to 100.0)
    # Uses 0.5% and 99.5% calibration bounds from training to prevent fabricated percentages
    score_min = preprocessor.get("score_min", -0.30)
    score_max = preprocessor.get("score_max", 0.20)

    # Safe linear normalization inverted so lower decision score = higher risk
    if score_max > score_min:
        normalized_pos = (raw_score - score_min) / (score_max - score_min)
        risk_score = (1.0 - normalized_pos) * 100.0
    else:
        risk_score = 50.0

    # Clamp strictly between 0.0 and 100.0
    risk_score = round(float(np.clip(risk_score, 0.0, 100.0)), 1)

    # 7. Categorical Risk Tier
    if risk_score < 40.0 and not is_anomaly:
        risk_level = "Low"
        prediction_label = "Normal"
    elif risk_score >= 70.0 or (is_anomaly and risk_score >= 60.0):
        risk_level = "High"
        prediction_label = "Suspected Fraud"
    else:
        risk_level = "Medium"
        prediction_label = "Suspected Fraud" if is_anomaly else "Normal (Monitor)"

    # 8. Explanations and educational disclaimer
    explanations = explain_flagged_factors(cleaned_input, risk_level)

    disclaimer = (
        "Important Note: Anomaly detection identifies unusual behavioral and statistical outliers; "
        "it highlights transactions requiring verification and does not constitute definitive proof of fraud."
    )

    return {
        "is_anomaly": is_anomaly,
        "prediction": prediction_label,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "anomaly_decision_score": round(raw_score, 4),
        "explanations": explanations,
        "input_data": cleaned_input,
        "disclaimer": disclaimer
    }
