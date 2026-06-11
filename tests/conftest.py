"""
Shared test fixtures for the fraud detection system test suite.
"""

import importlib
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))


@pytest.fixture
def sample_transaction_dict():
    """A valid transaction dictionary matching the Transaction schema."""
    return {
        "transaction_id": "TXN_TEST_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": "USER_1234",
        "amount_tnd": 2500.00,
        "governorate": "Tunis",
        "payment_method": "Flouci",
        "branch_id": "Tunis-GNC",
        "fraud_seed": False,
    }


@pytest.fixture
def high_risk_transaction_dict(sample_transaction_dict):
    """A transaction that should trigger high-risk alerts."""
    return {
        **sample_transaction_dict,
        "transaction_id": "TXN_HIGH_RISK_001",
        "amount_tnd": 12000.00,
        "fraud_seed": True,
    }


@pytest.fixture
def smurfing_transaction_dict(sample_transaction_dict):
    """A transaction in the smurfing detection range (1400-1500 TND)."""
    return {
        **sample_transaction_dict,
        "transaction_id": "TXN_SMURF_001",
        "amount_tnd": 1450.00,
        "payment_method": "Flouci",
    }


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database with the required schema."""
    db_path = tmp_path / "test_feedback.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            analyst_label TEXT,
            analyst_comment TEXT,
            analyst_id TEXT,
            branch_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS high_risk_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            user_id TEXT,
            amount_tnd REAL,
            governorate TEXT,
            payment_method TEXT,
            branch_id TEXT,
            timestamp TEXT,
            ml_probability REAL,
            sar_report TEXT,
            alert_type TEXT DEFAULT 'high_risk',
            shap_top5 TEXT,
            anomaly_score REAL,
            anomaly_model_version TEXT,
            ingestion_latency REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            version_id TEXT PRIMARY KEY,
            model_path TEXT NOT NULL,
            f1_score REAL,
            auc REAL,
            is_champion INTEGER DEFAULT 0,
            promoted_at DATETIME,
            training_samples_count INTEGER,
            feature_importance TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT,
            entity_id TEXT,
            action TEXT,
            user_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            previous_state TEXT,
            new_state TEXT
        )
    """
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def populated_db(tmp_db):
    """Database pre-populated with sample alerts and feedback."""
    conn = sqlite3.connect(str(tmp_db))
    cursor = conn.cursor()

    # Insert sample alerts
    alerts = [
        (
            "TXN_001",
            "USER_100",
            3000.0,
            "Tunis",
            "Flouci",
            "Tunis-GNC",
            datetime.now(timezone.utc).isoformat(),
            0.92,
            None,
            "high_risk",
        ),
        (
            "TXN_002",
            "USER_200",
            1450.0,
            "Sfax",
            "eDinar",
            "Sfax-Agency",
            datetime.now(timezone.utc).isoformat(),
            0.45,
            None,
            "random_sample",
        ),
        (
            "TXN_003",
            "USER_300",
            8500.0,
            "Sousse",
            "Konnect",
            "Sousse-Agency",
            datetime.now(timezone.utc).isoformat(),
            0.88,
            "SAR report text",
            "high_risk",
        ),
        (
            "TXN_004",
            "USER_100",
            500.0,
            "Ariana",
            "Carte Bancaire",
            "Ariana-Desk",
            datetime.now(timezone.utc).isoformat(),
            0.55,
            None,
            "uncertainty_sample",
        ),
    ]
    for a in alerts:
        cursor.execute(
            """
            INSERT INTO high_risk_alerts
            (transaction_id, user_id, amount_tnd, governorate, payment_method, branch_id,
             timestamp, ml_probability, sar_report, alert_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            a,
        )

    # Insert sample feedback
    feedback = [
        ("TXN_001", "Confirmed Fraud", "Obvious fraud pattern", "Tunis-GNC"),
        ("TXN_002", "False Positive", "Legitimate transfer", "Sfax-Agency"),
        ("TXN_003", "Confirmed Fraud", "Smurfing confirmed", "Sousse-Agency"),
    ]
    for f in feedback:
        cursor.execute(
            """
            INSERT INTO feedback_labels (transaction_id, analyst_label, analyst_comment, branch_id)
            VALUES (?, ?, ?, ?)
        """,
            f,
        )

    # Insert a champion model
    cursor.execute(
        """
        INSERT INTO model_registry
        (version_id, model_path, f1_score, auc, is_champion, promoted_at, training_samples_count, feature_importance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            "v1_test",
            "models/test_model",
            0.85,
            0.90,
            1,
            datetime.now().isoformat(),
            500,
            '[{"feature": "v_count", "score": 0.35}, {"feature": "avg_amount", "score": 0.28}, {"feature": "g_dist", "score": 0.20}]',
        ),
    )

    conn.commit()
    conn.close()
    return tmp_db


@pytest.fixture
def api_test_client(tmp_db, monkeypatch):
    """FastAPI TestClient with a temporary database."""
    monkeypatch.setenv("ADMIN_TOKEN", "test_admin_token")
    monkeypatch.setenv("ANALYST_TOKEN", "test_analyst_token")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_db.as_posix()}")
    monkeypatch.delenv("METRICS_TOKEN", raising=False)

    import shared.database as database_module

    importlib.reload(database_module)
    import dashboard.api as api_module

    importlib.reload(api_module)

    api_module.Base.metadata.create_all(bind=api_module.engine)

    from fastapi.testclient import TestClient

    client = TestClient(api_module.app)
    return client


@pytest.fixture
def admin_headers():
    """Authorization headers for admin role."""
    return {"Authorization": "Bearer test_admin_token"}


@pytest.fixture
def analyst_headers():
    """Authorization headers for analyst role."""
    return {"Authorization": "Bearer test_analyst_token"}
