# FraudShield AI – AI-Powered Financial Fraud Detection System

**College CIA-2 Project**  
*An intelligent, unsupervised machine learning system to detect anomalous and fraudulent financial transactions in real time.*

---

## 📌 Project Overview

Financial fraud is evolving rapidly with modern digital banking, UPI, and card payments. Traditional rule-based security systems rely on rigid static thresholds (e.g., flag transactions above \$10,000) which fraudsters easily circumvent, resulting in high false positives and missed frauds.

**FraudShield AI** addresses this challenge by employing **Machine Learning Anomaly Detection**. Rather than relying purely on static rules or requiring millions of strictly labeled fraud records, it uses the **Isolation Forest** algorithm to discover irregular behavioral patterns, sudden balance shifts, and atypical transaction amounts automatically.

---

## ⚙️ Technology Stack

| Component | Technology | Role & Purpose |
| :--- | :--- | :--- |
| **Backend** | Python Flask | Lightweight web framework exposing REST endpoints and serving the dashboard. |
| **Frontend** | HTML5, Vanilla CSS, JavaScript | Interactive financial dashboard displaying transaction metrics, fraud scores, and live alerts. |
| **Machine Learning** | Scikit-learn (Isolation Forest) | Unsupervised anomaly detection algorithm that separates outlier transactions from normal activity. |
| **Data Processing** | Pandas & NumPy | High-performance manipulation of transaction datasets and feature matrices. |
| **Model Persistence**| Joblib | Fast serialization and deserialization of the trained Scikit-learn model artifact. |
| **Database** | SQLite3 (Built-in Python) | Zero-configuration relational database storing transactions and evaluation logs without external database servers. |
| **Target OS** | Windows (PowerShell / CMD) | Fully configured for local Windows development and execution. |

---

## 🧠 Why Isolation Forest?

**Isolation Forest** is an unsupervised learning algorithm specifically built for anomaly detection:
1. **Core Concept**: Anomalous points (fraudulent transactions) are "few and different". Because of this, they are much easier to isolate than normal points.
2. **Path Length Principle**: In an ensemble of random decision trees (iTrees), anomalies are isolated very close to the tree roots (short path length), whereas normal points require many splits deeper in the tree (long path length).
3. **No Labeled Data Needed**: In the real world, 99.8% of transactions are legitimate and fraud tags are rare. Isolation Forest trains effectively on standard transaction distributions without needing heavily balanced labels.

---

## 📁 Project Directory Structure

```text
FraudShield-AI/
│
├── backend/                        # Flask Backend Application
│   ├── __init__.py                 # Marks backend as a Python package
│   ├── app.py                      # Flask server, route controllers & API endpoints
│   └── database.py                 # SQLite database connection & schema helper
│
├── frontend/                       # Web Presentation Layer
│   ├── templates/                  # Server-rendered HTML templates
│   │   └── index.html              # Main dashboard view
│   └── static/                     # Static assets served to the browser
│       ├── css/
│       │   └── style.css           # Styling, dark mode design tokens & layouts
│       └── js/
│           └── main.js             # Client-side JavaScript & health check logic
│
├── ml_model/                       # Machine Learning Architecture
│   ├── __init__.py                 # Marks ml_model as a Python package
│   ├── train.py                    # Script to preprocess data & train Isolation Forest
│   ├── predict.py                  # Module to load model & score new transactions
│   └── saved_models/               # Folder storing serialized model binaries
│       └── .gitkeep                # Keeps directory tracked in version control
│
├── database/                       # Database Storage & Migrations
│   ├── schema.sql                  # SQL DDL script defining tables & indexes
│   └── .gitkeep                    # Directory tracker (stores fraudshield.db at runtime)
│
├── data/                           # Datasets & Sample Records
│   └── sample_transactions.csv     # Sample dataset of financial transactions for training/testing
│
├── .gitignore                      # Excludes venv, cache files, and SQLite databases from git
├── requirements.txt                # Lean list of Python dependencies
├── run.py                          # Single-command launcher for Windows development
└── README.md                       # Comprehensive project documentation
```

---

## 📄 Detailed File & Folder Purpose Guide

### 1. Root Files
- **`run.py`**: The main entry point to start the Flask development server on Windows (`python run.py`). Sets up system paths cleanly.
- **`requirements.txt`**: Specifies exact dependencies needed: Flask, Scikit-learn, Pandas, NumPy, and Joblib.
- **`.gitignore`**: Prevents Python cache (`__pycache__`), virtual environment (`venv/`), and database files (`*.db`) from being accidentally committed.
- **`README.md`**: Project documentation, architectural explanations, and beginner setup guide.

### 2. `backend/` Folder
- **`__init__.py`**: Makes the backend folder importable across the project.
- **`app.py`**: Boots the Flask server, registers the static and template paths pointing to `frontend/`, provides the landing page route (`/`), and health check endpoint (`/api/health`).
- **`database.py`**: Centralized SQLite connection manager. Uses Python’s standard library `sqlite3` and executes `schema.sql`.

### 3. `frontend/` Folder
- **`templates/index.html`**: Semantic HTML5 layout featuring the dashboard header, setup confirmation banner, key metrics cards, and roadmap placeholders.
- **`static/css/style.css`**: Vanilla CSS stylesheet featuring modern dark-mode fintech color tokens, responsive card layouts, and typography.
- **`static/js/main.js`**: JavaScript code that automatically pings the backend `/api/health` endpoint on load to verify connectivity.

### 4. `ml_model/` Folder
- **`__init__.py`**: Exposes ML inference functions to the backend.
- **`train.py`**: Pipeline template that loads transaction data, extracts features (amounts, balance deltas), and fits the Scikit-learn Isolation Forest model.
- **`predict.py`**: Inference helper that takes transaction inputs, feeds them to the saved model, and outputs an anomaly decision (`Normal` vs. `Suspected Fraud`).
- **`saved_models/`**: Dedicated folder where `joblib` saves the trained model artifact (`isolation_forest.joblib`).

### 5. `database/` Folder
- **`schema.sql`**: The blueprint for the SQLite database. Creates the `transactions` table with columns for IDs, timestamps, amounts, balances, anomaly scores, and prediction flags.

### 6. `data/` Folder
- **`sample_transactions.csv`**: Realistic financial records (e.g., PAYMENT, CASH_OUT, TRANSFER) illustrating legitimate vs. fraudulent patterns.

---

## 🚀 Beginner Setup Guide for Windows

Follow these step-by-step instructions in **Windows PowerShell** or **Command Prompt**:

### Step 1: Open Terminal in Project Directory
Ensure your terminal is in the project root:
```powershell
cd c:\Users\RAJSHEKAR\FraudShield-AI
```

### Step 2: Create a Python Virtual Environment
Creating a virtual environment keeps your project dependencies isolated from the rest of your Windows system:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **In PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an Execution Policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, then activate).*
- **In Command Prompt (cmd.exe)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
*(Once activated, you will see `(venv)` at the beginning of your terminal line).*

### Step 4: Install Required Packages
Install the verified dependencies from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### Step 5: Run the Project
Launch the application:
```powershell
python run.py
```
Open your web browser and visit: **`http://127.0.0.1:5000`**

---

## 🔮 Next Development Steps (CIA-2 Roadmap)
1. **Train Model**: Implement the complete feature extraction and training pipeline in `ml_model/train.py`.
2. **SQLite Database**: Execute `database/schema.sql` to generate `fraudshield.db` and insert sample transactions.
3. **Inference API**: Connect `backend/app.py` to `ml_model/predict.py` to evaluate new transactions in real time.
4. **Interactive UI**: Build the fraud evaluation form and real-time transaction monitoring table on the frontend dashboard.
