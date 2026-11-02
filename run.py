
"""
FraudShield AI - Root Application Runner
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("FRAUDSHIELD AI - FINANCIAL FRAUD DETECTION SYSTEM")
    print(f"Server running on port: {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
