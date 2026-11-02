"""
FraudShield AI - Flask Application Backend
==========================================
College CIA-2 Project: AI-Powered Financial Fraud Detection System
Exposes REST API endpoints for real-time anomaly detection and serves
the modern fintech analytics dashboard.
"""

import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Ensure project root is in sys.path for clean cross-module imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import AI prediction module and database helpers
from ml_model.predict import predict_transaction
from backend.database import (
    init_db,
    insert_transaction,
    get_recent_transactions,
    get_dashboard_stats
)

TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)
app.config['SECRET_KEY'] = 'fraudshield-ai-secret-key-cia2'

# Ensure database is initialized on startup
try:
    init_db(seed_samples=True)
except Exception as e:
    print(f"[WARNING] Database initialization deferred: {e}")


@app.route("/")
def index():
    """
    Renders the main FraudShield AI fintech dashboard.
    """
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint returning system, model, and database status.
    """
    model_path = os.path.join(BASE_DIR, "ml_model", "saved_models", "isolation_forest.joblib")
    db_path = os.path.join(BASE_DIR, "database", "fraudshield.db")

    model_ready = os.path.exists(model_path)
    db_ready = os.path.exists(db_path)

    return jsonify({
        "status": "online",
        "service": "FraudShield AI Backend",
        "version": "2.0.0",
        "components": {
            "model_ready": model_ready,
            "database_ready": db_ready,
            "algorithm": "Isolation Forest (Scikit-Learn)"
        },
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Returns aggregated dashboard statistics from the SQLite database.
    """
    try:
        stats = get_dashboard_stats()
        return jsonify({"status": "success", "data": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    """
    Fetches recent transactions from the database.
    Optional query parameters:
      - filter: 'all', 'anomalies', 'normal' (default 'all')
      - limit: integer (default 50)
    """
    try:
        filter_type = request.args.get("filter", "all").lower()
        limit = int(request.args.get("limit", 50))
        limit = max(1, min(limit, 200))  # Bound between 1 and 200

        transactions = get_recent_transactions(limit=limit, filter_type=filter_type)
        return jsonify({
            "status": "success",
            "count": len(transactions),
            "filter": filter_type,
            "data": transactions
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def evaluate_transaction():
    """
    Accepts transaction details in JSON, executes the Isolation Forest anomaly
    detection model, persists the outcome in SQLite, and returns the verdict.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON payload. Please send application/json body."
            }), 400

        # Validate Amount
        if "amount" not in data:
            return jsonify({"status": "error", "message": "Field 'amount' is required."}), 400
        try:
            amount = float(data["amount"])
            if amount <= 0:
                return jsonify({"status": "error", "message": "Transaction amount must be greater than 0."}), 400
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "Amount must be a valid positive number."}), 400

        # Extract features with defaults
        transaction_type = str(data.get("transaction_type", "PAYMENT")).upper()
        
        try:
            hour = int(data.get("transaction_hour", datetime.now().hour))
            if not (0 <= hour <= 23):
                return jsonify({"status": "error", "message": "transaction_hour must be between 0 and 23."}), 400
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "transaction_hour must be an integer between 0 and 23."}), 400

        try:
            frequency = int(data.get("transaction_frequency", 1))
            if frequency < 1:
                return jsonify({"status": "error", "message": "transaction_frequency must be at least 1."}), 400
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "transaction_frequency must be an integer >= 1."}), 400

        try:
            is_unusual_location = 1 if int(data.get("is_unusual_location", 0)) == 1 else 0
        except (ValueError, TypeError):
            is_unusual_location = 0

        location_label = str(data.get("location_label") or ("Foreign / Unknown IP" if is_unusual_location == 1 else "Home / Verified Device"))

        # Generate unique transaction ID
        timestamp_now = datetime.now()
        timestamp_str = timestamp_now.strftime("%Y-%m-%d %H:%M:%S")
        custom_id = data.get("transaction_id")
        transaction_id = str(custom_id) if custom_id else f"TXN-{int(timestamp_now.timestamp() * 1000) % 100000000:08d}"

        # Run inference through the actual Isolation Forest model
        ai_input = {
            "amount": amount,
            "transaction_hour": hour,
            "transaction_frequency": frequency,
            "is_unusual_location": is_unusual_location
        }
        
        prediction_result = predict_transaction(ai_input)

        # Persist analyzed transaction into SQLite
        db_record = {
            "transaction_id": transaction_id,
            "timestamp": timestamp_str,
            "transaction_type": transaction_type,
            "amount": amount,
            "transaction_hour": hour,
            "transaction_frequency": frequency,
            "is_unusual_location": is_unusual_location,
            "location_label": location_label,
            "risk_score": prediction_result["risk_score"],
            "risk_level": prediction_result["risk_level"],
            "anomaly_decision_score": prediction_result["anomaly_decision_score"],
            "prediction": prediction_result["prediction"],
            "is_anomaly": 1 if prediction_result["is_anomaly"] else 0,
            "explanation": prediction_result["explanations"]
        }
        insert_transaction(db_record)

        # Build response payload
        response_payload = {
            "status": "success",
            "transaction_id": transaction_id,
            "timestamp": timestamp_str,
            "transaction_type": transaction_type,
            "amount": amount,
            "prediction": prediction_result["prediction"],
            "is_anomaly": prediction_result["is_anomaly"],
            "risk_score": prediction_result["risk_score"],
            "risk_level": prediction_result["risk_level"],
            "anomaly_decision_score": prediction_result["anomaly_decision_score"],
            "explanations": prediction_result["explanations"],
            "disclaimer": prediction_result["disclaimer"],
            "input_features": prediction_result["input_data"]
        }

        return jsonify(response_payload), 200

    except ValueError as ve:
        return jsonify({"status": "error", "message": f"Validation error: {str(ve)}"}), 400
    except FileNotFoundError as fe:
        return jsonify({"status": "error", "message": f"Model artifact error: {str(fe)}"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": f"Internal server error: {str(e)}"}), 500


@app.route("/api/seed", methods=["POST"])
def reseed_database():
    """
    Optional helper endpoint to re-seed sample transactions.
    """
    try:
        init_db(seed_samples=True)
        return jsonify({"status": "success", "message": "Database successfully synchronized."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
