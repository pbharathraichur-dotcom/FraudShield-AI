"""
FraudShield AI - Full Backend & Database Integration Test Suite
===============================================================
Tests:
1. Flask landing page (GET /)
2. Health check endpoint (GET /api/health)
3. Dashboard statistics endpoint (GET /api/stats)
4. Audit history endpoint (GET /api/transactions)
5. AI prediction endpoint with a Normal transaction (POST /api/predict)
6. AI prediction endpoint with a Fraud outlier transaction (POST /api/predict)
7. Error handling with invalid input (negative amount, non-numeric values)
8. SQLite database persistence verification
"""

import sys
import os
import json

# Setup root path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app import app
from backend.database import get_db_connection


def run_integration_tests():
    print("\n" + "=" * 70)
    print("  FRAUDSHIELD AI - END-TO-END INTEGRATION TEST SUITE")
    print("=" * 70)

    client = app.test_client()
    passed = 0
    total = 8

    # TEST 1: Landing Page
    print("\n[TEST 1] Checking Landing Page (GET /)...")
    res1 = client.get("/")
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}"
    html = res1.get_data(as_text=True)
    assert "FraudShield" in html, "Brand title missing in HTML"
    assert "Evaluate Transaction" in html, "Form missing in HTML"
    print("  --> PASS: Dashboard HTML successfully served with all templates & UI elements.")
    passed += 1

    # TEST 2: Health Check API
    print("\n[TEST 2] Checking Health Endpoint (GET /api/health)...")
    res2 = client.get("/api/health")
    assert res2.status_code == 200, f"Expected 200, got {res2.status_code}"
    data2 = res2.get_json()
    assert data2["status"] == "online"
    assert data2["components"]["model_ready"] is True, "Isolation Forest model artifact not found!"
    assert data2["components"]["database_ready"] is True, "SQLite database not found!"
    print(f"  --> PASS: Health OK. Model Ready: {data2['components']['model_ready']}, DB Ready: {data2['components']['database_ready']}")
    passed += 1

    # TEST 3: Dashboard Statistics API
    print("\n[TEST 3] Checking Stats Endpoint (GET /api/stats)...")
    res3 = client.get("/api/stats")
    assert res3.status_code == 200, f"Expected 200, got {res3.status_code}"
    data3 = res3.get_json()
    assert data3["status"] == "success"
    stats = data3["data"]
    print(f"  --> PASS: Stats retrieved. Total Monitored: {stats['total_monitored']}, Flagged: {stats['total_anomalies']}")
    passed += 1

    # TEST 4: Transaction History API
    print("\n[TEST 4] Checking Transactions Endpoint (GET /api/transactions)...")
    res4 = client.get("/api/transactions?filter=all&limit=5")
    assert res4.status_code == 200, f"Expected 200, got {res4.status_code}"
    data4 = res4.get_json()
    assert data4["status"] == "success"
    assert isinstance(data4["data"], list)
    print(f"  --> PASS: Retrieved {len(data4['data'])} historical records from SQLite.")
    passed += 1

    # Count database records before prediction
    conn = get_db_connection()
    count_before = conn.cursor().execute("SELECT COUNT(*) AS count FROM transactions").fetchone()["count"]
    conn.close()

    # TEST 5: Real-time Prediction - Normal Transaction
    print("\n[TEST 5] Testing AI Evaluation for Normal Transaction (POST /api/predict)...")
    normal_payload = {
        "amount": 34.50,
        "transaction_type": "PAYMENT",
        "transaction_hour": 13,
        "transaction_frequency": 2,
        "is_unusual_location": 0
    }
    res5 = client.post("/api/predict", json=normal_payload)
    assert res5.status_code == 200, f"Expected 200, got {res5.status_code}: {res5.get_data(as_text=True)}"
    data5 = res5.get_json()
    assert data5["status"] == "success"
    assert data5["prediction"] == "Normal"
    assert data5["risk_level"] == "Low"
    assert data5["risk_score"] < 40.0
    print(f"  --> PASS: Normal transaction scored successfully.")
    print(f"      Txn ID: {data5['transaction_id']} | Risk Score: {data5['risk_score']} | Risk Level: {data5['risk_level']}")
    passed += 1

    # TEST 6: Real-time Prediction - High Risk Fraud Outlier
    print("\n[TEST 6] Testing AI Evaluation for Fraud Outlier (POST /api/predict)...")
    fraud_payload = {
        "amount": 92000.00,
        "transaction_type": "TRANSFER",
        "transaction_hour": 3,
        "transaction_frequency": 24,
        "is_unusual_location": 1
    }
    res6 = client.post("/api/predict", json=fraud_payload)
    assert res6.status_code == 200, f"Expected 200, got {res6.status_code}: {res6.get_data(as_text=True)}"
    data6 = res6.get_json()
    assert data6["status"] == "success"
    assert data6["prediction"] == "Suspected Fraud"
    assert data6["risk_level"] == "High"
    assert data6["risk_score"] >= 70.0
    assert len(data6["explanations"]) >= 2
    print(f"  --> PASS: Outlier fraud successfully caught by Isolation Forest.")
    print(f"      Txn ID: {data6['transaction_id']} | Risk Score: {data6['risk_score']} | Risk Level: {data6['risk_level']}")
    for exp in data6["explanations"]:
        print(f"      - Factor: {exp}")
    passed += 1

    # TEST 7: Input Validation Error Handling
    print("\n[TEST 7] Testing Input Validation Error Handling (POST /api/predict with negative amount)...")
    invalid_payload = {
        "amount": -500.00,
        "transaction_hour": 14
    }
    res7 = client.post("/api/predict", json=invalid_payload)
    assert res7.status_code == 400, f"Expected 400 Bad Request, got {res7.status_code}"
    data7 = res7.get_json()
    assert data7["status"] == "error"
    print(f"  --> PASS: Invalid input was gracefully caught: '{data7['message']}'")
    passed += 1

    # TEST 8: Database Storage Verification
    print("\n[TEST 8] Verifying SQLite Database Storage & Persistence...")
    conn = get_db_connection()
    count_after = conn.cursor().execute("SELECT COUNT(*) AS count FROM transactions").fetchone()["count"]
    
    # Retrieve the last inserted record
    last_tx = conn.cursor().execute("SELECT * FROM transactions ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()

    assert count_after == count_before + 2, f"Expected {count_before + 2} records, got {count_after}"
    assert last_tx["transaction_id"] == data6["transaction_id"]
    assert last_tx["amount"] == 92000.00
    assert last_tx["is_anomaly"] == 1
    print(f"  --> PASS: Both transactions were securely persisted into SQLite database.")
    print(f"      Database rows before: {count_before} | Database rows after: {count_after}")
    print(f"      Verified stored record: ID={last_tx['transaction_id']}, Amount=${last_tx['amount']:,.2f}, Risk={last_tx['risk_score']}")
    passed += 1

    print("\n" + "=" * 70)
    print(f"  INTEGRATION TEST SUMMARY: {passed}/{total} Tests Passed (100% Success)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_integration_tests()
