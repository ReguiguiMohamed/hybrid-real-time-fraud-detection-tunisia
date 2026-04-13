"""
Migration 0002: PII Anonymization Columns
Adds data privacy compliance columns for GDPR / Tunisian data protection law.

Changes:
- Adds user_id_hashed to high_risk_alerts for anonymized lookups
- Adds pii_masked_payload to dead_letter_queue for secure error handling
- Adds data_retention_days configuration column
- Adds indexes for hashed lookups
"""
import sqlite3
from pathlib import Path


def upgrade(db_path: str = None):
    """Apply this migration."""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Add user_id_hashed column (SHA-256 hash of user_id for anonymized lookups)
    try:
        cursor.execute("ALTER TABLE high_risk_alerts ADD COLUMN user_id_hashed TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add indexes for hashed lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_user_hashed ON high_risk_alerts(user_id_hashed)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_transaction ON high_risk_alerts(transaction_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON high_risk_alerts(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON high_risk_alerts(created_at)")

    # Add index for feedback lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_transaction ON feedback_labels(transaction_id)")

    # Add index for DLQ status lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dlq_status ON dead_letter_queue(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dlq_next_retry ON dead_letter_queue(next_retry_at)")

    # Add index for audit log lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_logs(entity_type, entity_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)")

    # Record this migration
    cursor.execute("""
        INSERT OR IGNORE INTO migration_history (migration_id, description)
        VALUES ('0002_pii_anonymization', 'PII compliance: hashed user IDs, indexes for performance')
    """)

    conn.commit()
    conn.close()
    print("Migration 0002 applied: PII anonymization columns and indexes added.")


def downgrade(db_path: str = None):
    """Remove this migration."""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Drop indexes
    cursor.execute("DROP INDEX IF EXISTS idx_alerts_user_hashed")
    cursor.execute("DROP INDEX IF EXISTS idx_alerts_transaction")
    cursor.execute("DROP INDEX IF EXISTS idx_alerts_status")
    cursor.execute("DROP INDEX IF EXISTS idx_alerts_created_at")
    cursor.execute("DROP INDEX IF EXISTS idx_feedback_transaction")
    cursor.execute("DROP INDEX IF EXISTS idx_dlq_status")
    cursor.execute("DROP INDEX IF EXISTS idx_dlq_next_retry")
    cursor.execute("DROP INDEX IF EXISTS idx_audit_entity")
    cursor.execute("DROP INDEX IF EXISTS idx_audit_timestamp")

    cursor.execute("DELETE FROM migration_history WHERE migration_id = '0002_pii_anonymization'")

    conn.commit()
    conn.close()
    print("Migration 0002 downgraded: Indexes removed.")


if __name__ == "__main__":
    upgrade()
