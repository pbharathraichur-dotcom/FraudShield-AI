"""
FraudShield AI - Machine Learning Model Training Script
======================================================
College CIA-2 Project: AI-Powered Financial Fraud Detection System
Algorithm: Scikit-learn Isolation Forest (Unsupervised Anomaly Detection)

This script:
1. Loads sample transaction data from `data/sample_transactions.csv`.
2. Extracts core behavioral features:
   - `amount`: Financial transaction amount.
   - `transaction_hour`: Hour of the day (0 to 23).
   - `transaction_frequency`: Number of transactions by account in past 24 hours.
   - `is_unusual_location`: 0 for normal/recognized, 1 for foreign/unusual.
3. Generates synthetic transaction data to ensure statistical significance
   (since a 12-row dataset is insufficient for a 100-tree ensemble).
4. Cleans and preprocesses data (handling missing/invalid values and scaling).
5. Fits the Isolation Forest model.
6. Calibrates decision scores and saves the model and preprocessor artifacts
   in `ml_model/saved_models/`.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Setup absolute project paths for Windows compatibility
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_transactions.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "ml_model", "saved_models")
MODEL_FILE = os.path.join(SAVED_MODELS_DIR, "isolation_forest.joblib")
PREPROCESSOR_FILE = os.path.join(SAVED_MODELS_DIR, "preprocessor.joblib")

# Core feature names used for training and prediction
FEATURE_COLUMNS = [
    "amount",
    "transaction_hour",
    "transaction_frequency",
    "is_unusual_location"
]


def load_sample_data(filepath):
    """
    Load data from CSV and extract core features.
    """
    if not os.path.exists(filepath):
        print(f"[WARNING] Sample file not found at {filepath}. Starting with synthetic data.")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} records from {filepath}")

    # Extract transaction_hour from timestamp if timestamp exists
    if "timestamp" in df.columns:
        df["transaction_hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    elif "transaction_hour" not in df.columns:
        df["transaction_hour"] = 12  # Default midday

    # Ensure required features exist or provide fallback defaults
    if "transaction_frequency" not in df.columns:
        df["transaction_frequency"] = 2
    if "is_unusual_location" not in df.columns:
        df["is_unusual_location"] = 0

    return df


def generate_synthetic_transactions(n_samples=1500, anomaly_ratio=0.08, random_state=42):
    """
    Generate realistic synthetic financial transactions.
    
    Why synthetic data is needed:
    Isolation Forest relies on building 100 random decision trees with sub-sampling.
    A dataset of 12 rows is mathematically too small for the trees to isolate patterns.
    Augmenting with realistic normal and anomalous distributions provides statistically
    valid decision boundaries.
    """
    np.random.seed(random_state)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # 1. Normal transactions:
    # - Amounts: mostly modest ($10 to $500, lognormal distribution)
    # - Hours: daytime/evening (8 AM to 10 PM)
    # - Frequency: 1 to 5 transactions per day
    # - Location: almost always recognized (98% = 0)
    normal_amounts = np.random.lognormal(mean=4.2, sigma=0.9, size=n_normal)
    normal_amounts = np.clip(normal_amounts, 5.0, 1800.0)

    normal_hours = np.random.normal(loc=14.0, scale=3.5, size=n_normal)
    normal_hours = np.clip(normal_hours, 6, 23).astype(int)

    normal_freq = np.random.poisson(lam=2.5, size=n_normal)
    normal_freq = np.clip(normal_freq, 1, 6)

    normal_locations = np.random.choice([0, 1], size=n_normal, p=[0.98, 0.02])

    df_normal = pd.DataFrame({
        "amount": normal_amounts,
        "transaction_hour": normal_hours,
        "transaction_frequency": normal_freq,
        "is_unusual_location": normal_locations
    })

    # 2. Anomalous transactions (Simulated fraudulent behavior):
    # - Amounts: sudden large values ($5,000 to $90,000)
    # - Hours: unusual late-night hours (1 AM to 4 AM)
    # - Frequency: high-velocity bursts (12 to 35 transactions/day)
    # - Location: high chance of unknown location (75% = 1)
    anomaly_amounts = np.random.uniform(5000.0, 95000.0, size=n_anomalies)
    anomaly_hours = np.random.choice([0, 1, 2, 3, 4, 5], size=n_anomalies)
    anomaly_freq = np.random.randint(12, 35, size=n_anomalies)
    anomaly_locations = np.random.choice([0, 1], size=n_anomalies, p=[0.25, 0.75])

    df_anomalies = pd.DataFrame({
        "amount": anomaly_amounts,
        "transaction_hour": anomaly_hours,
        "transaction_frequency": anomaly_freq,
        "is_unusual_location": anomaly_locations
    })

    df_synthetic = pd.concat([df_normal, df_anomalies], ignore_index=True)
    return df_synthetic.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def clean_and_prepare_features(df):
    """
    Handle missing values, invalid types, and non-numeric entries.
    """
    df_clean = df.copy()

    for col in FEATURE_COLUMNS:
        # Convert to numeric, replacing invalid strings with NaN
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Impute missing values with sensible defaults
    df_clean["amount"] = df_clean["amount"].fillna(df_clean["amount"].median())
    df_clean["amount"] = df_clean["amount"].apply(lambda x: max(0.01, float(x)))

    df_clean["transaction_hour"] = df_clean["transaction_hour"].fillna(12)
    df_clean["transaction_hour"] = df_clean["transaction_hour"].apply(lambda x: int(np.clip(x, 0, 23)))

    df_clean["transaction_frequency"] = df_clean["transaction_frequency"].fillna(1)
    df_clean["transaction_frequency"] = df_clean["transaction_frequency"].apply(lambda x: max(1, int(x)))

    df_clean["is_unusual_location"] = df_clean["is_unusual_location"].fillna(0)
    df_clean["is_unusual_location"] = df_clean["is_unusual_location"].apply(lambda x: 1 if int(x) == 1 else 0)

    return df_clean[FEATURE_COLUMNS]


def train_isolation_forest():
    """
    Main training function to fit and export the Isolation Forest model.
    """
    print("=" * 65)
    print("  FRAUDSHIELD AI - ISOLATION FOREST MODEL TRAINING")
    print("=" * 65)

    # Ensure output directory exists
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # 1. Load sample dataset
    df_sample = load_sample_data(DATA_PATH)
    sample_features = clean_and_prepare_features(df_sample) if not df_sample.empty else pd.DataFrame()

    # 2. Generate synthetic transactions for robust training
    print("[INFO] Generating realistic synthetic transaction distributions...")
    df_synthetic = generate_synthetic_transactions(n_samples=1500, anomaly_ratio=0.08, random_state=42)

    # 3. Combine real sample data with synthetic data
    if not sample_features.empty:
        df_training = pd.concat([sample_features, df_synthetic], ignore_index=True)
    else:
        df_training = df_synthetic

    df_training = clean_and_prepare_features(df_training)
    print(f"[INFO] Total training records prepared: {len(df_training)}")
    print(f"[INFO] Features: {', '.join(FEATURE_COLUMNS)}")

    # 4. Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_training[FEATURE_COLUMNS])

    # 5. Initialize and train Isolation Forest
    # Contamination = expected proportion of outliers in the data (~8%)
    contamination_rate = 0.08
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=contamination_rate,
        random_state=42,
        n_jobs=-1
    )

    print(f"[INFO] Fitting Isolation Forest with {iso_forest.n_estimators} trees...")
    iso_forest.fit(X_scaled)

    # 6. Evaluate decision scores across training set for calibration
    # Scikit-learn's decision_function returns:
    # - Negative values for anomalies (outliers)
    # - Positive values for normal observations (inliers)
    decision_scores = iso_forest.decision_function(X_scaled)
    preds = iso_forest.predict(X_scaled)

    n_flagged = int(np.sum(preds == -1))
    n_normal = int(np.sum(preds == 1))

    # Determine 1st and 99th percentiles for robust 0-100 score normalization
    score_min = float(np.percentile(decision_scores, 0.5))
    score_max = float(np.percentile(decision_scores, 99.5))

    print("-" * 65)
    print(f"[RESULTS] Model Training Summary:")
    print(f"  * Normal transactions detected:    {n_normal} ({(n_normal/len(preds))*100:.1f}%)")
    print(f"  * Anomalous transactions flagged:  {n_flagged} ({(n_flagged/len(preds))*100:.1f}%)")
    print(f"  * Decision score range:           [{decision_scores.min():.4f}, {decision_scores.max():.4f}]")
    print(f"  * Calibration scale (0.5% - 99.5%): [{score_min:.4f}, {score_max:.4f}]")

    # 7. Save model and preprocessor artifacts
    preprocessor = {
        "features": FEATURE_COLUMNS,
        "scaler": scaler,
        "score_min": score_min,
        "score_max": score_max,
        "contamination": contamination_rate,
        "train_samples_count": len(df_training)
    }

    joblib.dump(iso_forest, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    print("-" * 65)
    print(f"[SUCCESS] Model artifact saved to:        {MODEL_FILE}")
    print(f"[SUCCESS] Preprocessor metadata saved to:  {PREPROCESSOR_FILE}")
    print("=" * 65 + "\n")

    return {
        "model_file": MODEL_FILE,
        "preprocessor_file": PREPROCESSOR_FILE,
        "records": len(df_training),
        "anomalies_flagged": n_flagged
    }


if __name__ == "__main__":
    train_isolation_forest()
