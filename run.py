"""
FraudShield AI - Root Application Runner
Run this file to launch the complete Flask web server on Windows.

Usage:
    python run.py
"""

import sys
import os

# Add root directory to python path for clean package imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FRAUDSHIELD AI - FINANCIAL FRAUD DETECTION SYSTEM")
    print("  College CIA-2 Project")
    print("=" * 60)
    print("  Server running locally at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60 + "\n")
    
    app.run(host="127.0.0.1", port=5000, debug=True)
