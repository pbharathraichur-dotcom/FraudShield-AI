"""
FraudShield AI - Database Helper Module
=======================================
Manages SQLite database connections, schema migrations, transaction persistence,
and real-time statistics aggregation for the dashboard.
"""

import sqlite3
import os
import sys
import json
from datetime import datetime

# Windows-compatible absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "database", "fraudshield.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "database", "schema.sql")
DATA_CSV_PATH = os.path.join(BASE_DIR, "data", "sample_transactions.csv")


def get_db_connection():
    """
    Establish a connection to SQLite database.
    Returns sqlite3.Connection with Row factory enabled for key-value access.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(seed_samples=True):
    """
    Initialize SQLite database tables using schema.sql.
    Seeds sample data on first run if database is empty.
    """
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")

    conn = get_db_connection()
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()

    # Check if table already has rows
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS count FROM transactions")
    row_count = cursor.fetchone()["count"]

    conn.close()
    print(f"[INFO] SQLite database initialized at: {DB_PATH}")

    if seed_samples and row_count == 0:
        seed_sample_records()


def seed_sample_records():
    """
    Seed initial historical transactions from sample_transactions.csv
    by running them through the actual Isolation Forest model.
    """
    if not os.path.exists(DATA_CSV_PATH):
        return

    try:
        import pandas as pd
        from ml_model.predict import predict_transaction

        df = pd.read_csv(DATA_CSV_PATH)
        print(f"[INFO] Seeding {len(df)} initial transactions into database...")

        for _, row in df.iterrows():
            tx_id = str(row.get("transaction_id", f"TXN-{int(datetime.now().timestamp())}"))
            tx_type = str(row.get("type", "PAYMENT"))
            amount = float(row.get("amount", 100.0))
            
            # Extract hour
            hour = 12
            if "timestamp" in row and pd.notnull(row["timestamp"]):
                try:
                    hour = pd.to_datetime(row["timestamp"]).hour
                except Exception:
                    hour = 12
            elif "transaction_hour" in row:
                hour = int(row["transaction_hour"])

            freq = int(row.get("transaction_frequency", 2))
            is_unusual = int(row.get("is_unusual_location", 0))
            loc_label = "Foreign / Unknown IP" if is_unusual == 1 else "Home / Verified Device"

            # Evaluate with actual AI model
            ai_res = predict_transaction({
                "amount": amount,
                "transaction_hour": hour,
                "transaction_frequency": freq,
                "is_unusual_location": is_unusual
            })

            insert_transaction({
                "transaction_id": tx_id,
                "timestamp": str(row.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                "transaction_type": tx_type,
                "amount": amount,
                "transaction_hour": hour,
                "transaction_frequency": freq,
                "is_unusual_location": is_unusual,
                "location_label": loc_label,
                "risk_score": ai_res["risk_score"],
                "risk_level": ai_res["risk_level"],
                "anomaly_decision_score": ai_res["anomaly_decision_score"],
                "prediction": ai_res["prediction"],
                "is_anomaly": 1 if ai_res["is_anomaly"] else 0,
                "explanation": json.dumps(ai_res["explanations"])
            })

        print("[SUCCESS] Initial sample transactions seeded with actual model predictions.")
    except Exception as e:
        print(f"[WARNING] Could not seed sample data (Model may need training first): {e}")


def insert_transaction(data):
    """
    Insert a newly evaluated transaction record into the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    sql = """
    INSERT INTO transactions (
        transaction_id, timestamp, transaction_type, amount,
        transaction_hour, transaction_frequency, is_unusual_location,
        location_label, risk_score, risk_level, anomaly_decision_score,
        prediction, is_anomaly, explanation
    ) VALUES (
        :transaction_id, :timestamp, :transaction_type, :amount,
        :transaction_hour, :transaction_frequency, :is_unusual_location,
        :location_label, :risk_score, :risk_level, :anomaly_decision_score,
        :prediction, :is_anomaly, :explanation
    )
    """

    # Ensure required fields and formats
    tx_timestamp = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = {
        "transaction_id": data["transaction_id"],
        "timestamp": tx_timestamp,
        "transaction_type": data.get("transaction_type", "PAYMENT"),
        "amount": float(data["amount"]),
        "transaction_hour": int(data.get("transaction_hour", 12)),
        "transaction_frequency": int(data.get("transaction_frequency", 1)),
        "is_unusual_location": int(data.get("is_unusual_location", 0)),
        "location_label": data.get("location_label", "Local / Domestic"),
        "risk_score": float(data.get("risk_score", 0.0)),
        "risk_level": data.get("risk_level", "Low"),
        "anomaly_decision_score": float(data.get("anomaly_decision_score", 0.0)),
        "prediction": data.get("prediction", "Normal"),
        "is_anomaly": int(data.get("is_anomaly", 0)),
        "explanation": data.get("explanation") if isinstance(data.get("explanation"), str) else json.dumps(data.get("explanation", []))
    }

    cursor.execute(sql, params)
    conn.commit()
    conn.close()


def get_recent_transactions(limit=50, filter_type="all"):
    """
    Fetch recent transactions from database with optional filtering.
    filter_type: 'all', 'anomalies', 'normal'
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    base_query = "SELECT * FROM transactions"
    conditions = []
    params = []

    if filter_type == "anomalies":
        conditions.append("is_anomaly = 1")
    elif filter_type == "normal":
        conditions.append("is_anomaly = 0")

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(base_query, params)
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        item = dict(r)
        # Parse JSON explanations if serialized
        if item.get("explanation"):
            try:
                item["explanation"] = json.loads(item["explanation"])
            except Exception:
                pass
        result.append(item)

    return result


def get_dashboard_stats():
    """
    Compute aggregate dashboard metrics from SQLite database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            COUNT(*) AS total_transactions,
            COALESCE(SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END), 0) AS total_anomalies,
            COALESCE(SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END), 0) AS high_risk_count,
            COALESCE(SUM(CASE WHEN risk_level = 'Medium' THEN 1 ELSE 0 END), 0) AS medium_risk_count,
            COALESCE(SUM(CASE WHEN risk_level = 'Low' THEN 1 ELSE 0 END), 0) AS low_risk_count,
            COALESCE(AVG(risk_score), 0.0) AS avg_risk_score,
            COALESCE(SUM(amount), 0.0) AS total_volume,
            COALESCE(SUM(CASE WHEN is_anomaly = 1 THEN amount ELSE 0 END), 0.0) AS fraud_volume
        FROM transactions
    """)
    stats_row = cursor.fetchone()

    # Risk level distribution for chart
    risk_distribution = {
        "low": stats_row["low_risk_count"],
        "medium": stats_row["medium_risk_count"],
        "high": stats_row["high_risk_count"]
    }

    # Hourly distribution of anomalies
    cursor.execute("""
        SELECT transaction_hour, COUNT(*) as count 
        FROM transactions 
        WHERE is_anomaly = 1 
        GROUP BY transaction_hour 
        ORDER BY transaction_hour ASC
    """)
    hourly_rows = cursor.fetchall()
    hourly_anomalies = {row["transaction_hour"]: row["count"] for row in hourly_rows}

    conn.close()

    total_tx = stats_row["total_transactions"]
    anomaly_rate = round((stats_row["total_anomalies"] / total_tx * 100), 1) if total_tx > 0 else 0.0

    return {
        "total_monitored": total_tx,
        "total_anomalies": stats_row["total_anomalies"],
        "normal_count": total_tx - stats_row["total_anomalies"],
        "high_risk_count": stats_row["high_risk_count"],
        "medium_risk_count": stats_row["medium_risk_count"],
        "low_risk_count": stats_row["low_risk_count"],
        "anomaly_rate": anomaly_rate,
        "avg_risk_score": round(stats_row["avg_risk_score"], 1),
        "total_volume": round(stats_row["total_volume"], 2),
        "fraud_volume": round(stats_row["fraud_volume"], 2),
        "risk_distribution": risk_distribution,
        "hourly_anomalies": hourly_anomalies
    }


if __name__ == "__main__":
    init_db(seed_samples=True)
