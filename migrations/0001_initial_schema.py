"""
Migration 0001: Initial Schema
Creates the foundational tables for the fraud detection pipeline.

This is the baseline migration that establishes:
- high_risk_alerts: Stores all fraud alerts from the pipeline
- feedback_labels: Analyst feedback for model retraining
- model_registry: Champion-challenger model versioning
- audit_logs: Compliance audit trail for all administrative actions
- dead_letter_queue: Failed alerts for retry processing
"""
import sqlite3
from pathlib import Path


def upgrade(db_path: str = None):
    """Apply this migration."""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Enable WAL mode for concurrent reads
    cursor.execute("PRAGMA journal_mode=WAL")

    # 1. High Risk Alerts Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS high_risk_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            user_id TEXT NOT NULL,
            amount_tnd REAL NOT NULL,
            governorate TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            branch_id TEXT,
            timestamp TEXT NOT NULL,
            ml_probability REAL DEFAULT 0.0,
            alert_type TEXT NOT NULL DEFAULT 'high_risk',
            sar_report TEXT,
            ingestion_latency REAL DEFAULT 0.0,
            status TEXT DEFAULT 'pending',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            risk_factors TEXT,
            user_id_hashed TEXT
        )
    """)

    # 2. Feedback Labels Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            analyst_label TEXT NOT NULL,
            analyst_comment TEXT,
            analyst_id TEXT,
            branch_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (transaction_id) REFERENCES high_risk_alerts(transaction_id)
        )
    """)

    # 3. Model Registry Table
    cursor.execute("""
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
    """)

    # 4. Audit Logs Table
    cursor.execute("""
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
    """)

    # 5. Dead Letter Queue Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dead_letter_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            error_code TEXT,
            error_message TEXT,
            alert_payload TEXT,
            status TEXT DEFAULT 'PENDING',
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            next_retry_at DATETIME
        )
    """)

    # 6. Migration tracking table (for Alembic-style tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS migration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            migration_id TEXT NOT NULL UNIQUE,
            description TEXT,
            applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            checksum TEXT
        )
    """)

    # Record this migration
    cursor.execute("""
        INSERT OR IGNORE INTO migration_history (migration_id, description)
        VALUES ('0001_initial_schema', 'Initial schema: alerts, feedback, model_registry, audit_logs, dlq')
    """)

    conn.commit()
    conn.close()
    print("Migration 0001 applied: Initial schema created.")


def downgrade(db_path: str = None):
    """Remove this migration (drops all tables)."""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # SQLite doesn't support DROP TABLE IF EXISTS CASCADE, so drop in dependency order
    cursor.execute("DROP TABLE IF EXISTS feedback_labels")
    cursor.execute("DROP TABLE IF EXISTS dead_letter_queue")
    cursor.execute("DROP TABLE IF EXISTS audit_logs")
    cursor.execute("DROP TABLE IF EXISTS model_registry")
    cursor.execute("DROP TABLE IF EXISTS high_risk_alerts")
    cursor.execute("DELETE FROM migration_history WHERE migration_id = '0001_initial_schema'")

    conn.commit()
    conn.close()
    print("Migration 0001 downgraded: All tables dropped.")


if __name__ == "__main__":
    upgrade()
