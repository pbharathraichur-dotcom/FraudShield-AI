"""
FraudShield AI - Machine Learning Model Verification & Test Suite
==================================================================
Runs realistic test cases through `ml_model/predict.py` to evaluate:
1. Everyday legitimate transactions (Low risk)
2. Borderline / elevated transactions (Medium risk)
3. Severe anomalies and suspected fraud (High risk)
4. Input validation and error handling
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model.predict import predict_transaction


def run_tests():
    print("\n" + "=" * 70)
    print("  FRAUDSHIELD AI - MODEL EVALUATION & INFERENCE TEST SUITE")
    print("=" * 70)

    test_cases = [
        {
            "name": "Case 1: Standard Everyday Purchase (Lunch / Retail)",
            "data": {
                "amount": 28.50,
                "transaction_hour": 13,
                "transaction_frequency": 2,
                "is_unusual_location": 0
            },
            "expected_risk": "Low"
        },
        {
            "name": "Case 2: Moderate Evening Online Shopping",
            "data": {
                "amount": 420.00,
                "transaction_hour": 19,
                "transaction_frequency": 3,
                "is_unusual_location": 0
            },
            "expected_risk": "Low"
        },
        {
            "name": "Case 3: Borderline High Amount During Normal Hours",
            "data": {
                "amount": 3200.00,
                "transaction_hour": 11,
                "transaction_frequency": 7,
                "is_unusual_location": 0
            },
            "expected_risk": "Medium"
        },
        {
            "name": "Case 4: Overnight Large Cash-Out with Foreign IP (Classic Fraud)",
            "data": {
                "amount": 88000.00,
                "transaction_hour": 3,
                "transaction_frequency": 22,
                "is_unusual_location": 1
            },
            "expected_risk": "High"
        },
        {
            "name": "Case 5: High Velocity Burst at 2 AM (Card Testing / Bot)",
            "data": {
                "amount": 95.00,
                "transaction_hour": 2,
                "transaction_frequency": 32,
                "is_unusual_location": 1
            },
            "expected_risk": "High"
        }
    ]

    passed_count = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Running Test {i}: {test['name']} ---")
        data = test["data"]
        print(f"  Inputs: Amount=${data['amount']:,.2f} | Hour={data['transaction_hour']:02d}:00 | Freq={data['transaction_frequency']} | Unusual Loc={data['is_unusual_location']}")

        try:
            result = predict_transaction(data)
            
            # Format and display results
            print(f"  * Prediction:             {result['prediction']}")
            print(f"  * Risk Level:             {result['risk_level']}")
            print(f"  * Calibrated Risk Score:  {result['risk_score']} / 100")
            print(f"  * Anomaly Decision Score: {result['anomaly_decision_score']}")
            print(f"  * Explanations:")
            for reason in result["explanations"]:
                print(f"      - {reason}")
            
            passed_count += 1

        except Exception as e:
            print(f"  [ERROR] Prediction failed with exception: {e}")

    # Test Input Validation
    print("\n--- Running Test 6: Input Validation (Negative Amount Error Handling) ---")
    try:
        predict_transaction({"amount": -50.0, "transaction_hour": 12})
        print("  [FAIL] Expected ValueError for negative amount, but none was raised.")
    except ValueError as ve:
        print(f"  [PASS] Correctly caught invalid input: {ve}")
        passed_count += 1
    except Exception as e:
        print(f"  [FAIL] Unexpected exception type: {e}")

    print("\n" + "=" * 70)
    print(f"  TEST RESULTS SUMMARY: {passed_count}/{len(test_cases) + 1} Tests Executed Successfully")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_tests()
