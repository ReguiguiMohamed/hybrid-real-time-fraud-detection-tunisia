"""
Migration 0003: Dynamic Rules Engine Tables
Replaces hard-coded risk_config.py with database-driven rules.

Changes:
- Creates risk_rules table for dynamic weight/threshold configuration
- Creates governorate_risk_profiles for geographic risk scoring
- Creates d17_rules for e-wallet specific regulations
- Creates rule_change_log for audit trail of rule modifications
"""
import sqlite3
from pathlib import Path


def upgrade(db_path: str = None):
    """Apply this migration."""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. Dynamic Risk Rules Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_name TEXT NOT NULL UNIQUE,
            rule_type TEXT NOT NULL,
            weight REAL DEFAULT 0.0,
            threshold REAL DEFAULT 0.0,
            is_active INTEGER DEFAULT 1,
            description TEXT,
            created_by TEXT DEFAULT 'system',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert default rules (matching current risk_config.py)
    cursor.execute("""
        INSERT OR IGNORE INTO risk_rules (rule_name, rule_type, weight, threshold, description)
        VALUES
            ('velocity', 'frequency_weight', 0.3, 3.0, 'Transaction frequency in 5-min window'),
            ('travel', 'geo_anomaly', 0.3, 1.0, 'Distinct governorates in window (impossible travel)'),
            ('high_value', 'amount_threshold', 0.2, 5000.0, 'High value transaction threshold (TND)'),
            ('d17_limit', 'ewallet_limit', 0.2, 2000.0, 'D17/e-wallet smurfing detection threshold')
    """)

    # 2. Governorate Risk Profiles
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS governorate_risk_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            governorate_name TEXT NOT NULL UNIQUE,
            risk_multiplier REAL DEFAULT 1.0,
            is_cbdc_pilot INTEGER DEFAULT 0,
            is_high_risk_zone INTEGER DEFAULT 0,
            notes TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert all 24 Tunisian governorates
    governorates = [
        ("Tunis", 1.2, 1, 0, "Capital, CBDC pilot zone"),
        ("Sfax", 1.1, 1, 0, "Major commercial hub, CBDC pilot zone"),
        ("Sousse", 1.0, 0, 0, "Tourist hub"),
        ("Ariana", 1.0, 0, 0, ""),
        ("Bizerte", 1.0, 0, 0, ""),
        ("Gabes", 1.0, 0, 0, ""),
        ("Kairouan", 1.1, 0, 1, "Historical risk factor"),
        ("Manouba", 1.0, 0, 0, ""),
        ("Ben Arous", 1.0, 0, 0, ""),
        ("Nabeul", 1.0, 0, 0, "Tourist area"),
        ("Zaghouan", 1.0, 0, 0, ""),
        ("Monastir", 1.0, 0, 0, "Tourist area"),
        ("Mahdia", 1.0, 0, 0, ""),
        ("Kasserine", 1.2, 0, 1, "Border region, elevated risk"),
        ("Sidi Bouzid", 1.1, 0, 1, "Border region, elevated risk"),
        ("Gafsa", 1.1, 0, 1, "Mining region, elevated risk"),
        ("Tozeur", 1.0, 0, 0, ""),
        ("Kebili", 1.0, 0, 0, ""),
        ("Medenine", 1.1, 0, 1, "Border region, cross-border trade"),
        ("Tataouine", 1.2, 0, 1, "Southern border, elevated risk"),
        ("Jendouba", 1.0, 0, 0, ""),
        ("Beja", 1.0, 0, 0, ""),
        ("Le Kef", 1.1, 0, 1, "Border region, elevated risk"),
        ("Siliana", 1.0, 0, 0, ""),
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO governorate_risk_profiles
        (governorate_name, risk_multiplier, is_cbdc_pilot, is_high_risk_zone, notes)
        VALUES (?, ?, ?, ?, ?)
    """, governorates)

    # 3. D17 / E-Wallet Specific Rules
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS d17_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_name TEXT NOT NULL UNIQUE,
            ewallet_provider TEXT NOT NULL,
            threshold_amount REAL,
            velocity_limit INTEGER,
            window_minutes INTEGER,
            risk_boost REAL DEFAULT 0.0,
            is_active INTEGER DEFAULT 1,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert default D17 rules
    cursor.execute("""
        INSERT OR IGNORE INTO d17_rules
        (rule_name, ewallet_provider, threshold_amount, velocity_limit, window_minutes, risk_boost, description)
        VALUES
            ('flouci_high_value', 'Flouci', 2000.0, NULL, NULL, 0.2, 'Flouci transactions >2000 TND risk boost'),
            ('flouci_velocity', 'Flouci', NULL, 5, 5, 0.0, 'Flouci max 5 transactions per 5-min window'),
            ('d17_soft_limit_audit', 'Flouci', 1500.0, NULL, NULL, 0.0, 'D17 soft limit: triggers audit flag'),
            ('smurfing_range', 'Flouci', 1400.0, NULL, NULL, 0.0, 'Lower bound of smurfing detection range (1400-1500)')
    """)

    # 4. Rule Change Log (Audit Trail)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rule_change_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_table TEXT NOT NULL,
            rule_id INTEGER,
            rule_name TEXT,
            change_type TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            changed_by TEXT DEFAULT 'system',
            changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            reason TEXT
        )
    """)

    # Record this migration
    cursor.execute("""
        INSERT OR IGNORE INTO migration_history (migration_id, description)
        VALUES ('0003_rules_engine', 'Dynamic rules engine: risk_rules, governorate_profiles, d17_rules, change_log')
    """)

    conn.commit()
    conn.close()
    print("Migration 0003 applied: Dynamic rules engine tables created with default values.")


def downgrade(db_path: str = None):
    """Remove this migration."""
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS rule_change_log")
    cursor.execute("DROP TABLE IF EXISTS d17_rules")
    cursor.execute("DROP TABLE IF EXISTS governorate_risk_profiles")
    cursor.execute("DROP TABLE IF EXISTS risk_rules")

    cursor.execute("DELETE FROM migration_history WHERE migration_id = '0003_rules_engine'")

    conn.commit()
    conn.close()
    print("Migration 0003 downgraded: Rules engine tables dropped.")


if __name__ == "__main__":
    upgrade()
