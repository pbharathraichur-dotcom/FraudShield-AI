-- FraudShield AI - Database Schema for SQLite
-- Stores incoming transactions, AI Isolation Forest anomaly scores, risk levels, and detection logs.

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT UNIQUE NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    transaction_type TEXT NOT NULL DEFAULT 'PAYMENT',
    amount REAL NOT NULL,
    transaction_hour INTEGER NOT NULL DEFAULT 12,
    transaction_frequency INTEGER NOT NULL DEFAULT 1,
    is_unusual_location INTEGER NOT NULL DEFAULT 0,
    location_label TEXT DEFAULT 'Local / Domestic',
    risk_score REAL NOT NULL DEFAULT 0.0,
    risk_level TEXT NOT NULL DEFAULT 'Low',
    anomaly_decision_score REAL DEFAULT 0.0,
    prediction TEXT NOT NULL DEFAULT 'Normal',
    is_anomaly INTEGER DEFAULT 0,
    explanation TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transactions_txnid ON transactions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_transactions_created ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_transactions_anomaly ON transactions(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_transactions_risk ON transactions(risk_level);
