"""
Migration 0004: Finance Law 2026 — Rule Updates

Changes:
- Removes 'high_value' rule's cash-cap semantic (TND 5,000 cap repealed by Finance Law 2026)
  and replaces it with a general large-transaction monitoring threshold (TND 15,000).
- Adds velocity-based smurfing rules that do not depend on a hard cash ceiling:
    smurfing_velocity_unit_cap   — per-transaction amount ceiling for smurfing pattern
    smurfing_velocity_agg_min    — minimum window-aggregate to trigger smurfing flag
    smurfing_velocity_min_count  — minimum transaction count in window to flag
- Logs all changes in rule_change_log with regulatory justification.
"""
import sqlite3
from pathlib import Path


_MIGRATION_ID = "0004_finance_law_2026"
_CHANGED_BY = "system/migration"
_REASON = (
    "Finance Law 2026 (enacted Dec 2025) repealed the TND 5,000 hard cash-payment ceiling. "
    "Threshold updated from 5,000 to 15,000 TND for general large-tx AML monitoring. "
    "Velocity-based smurfing rules added as the primary structuring-detection mechanism."
)


def upgrade(db_path: str = None):
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. Fetch old threshold for audit log
    cursor.execute(
        "SELECT threshold, description FROM risk_rules WHERE rule_name = 'high_value'"
    )
    row = cursor.fetchone()
    old_threshold = row[0] if row else 5000.0
    old_desc = row[1] if row else ""

    # 2. Update high_value threshold: cash-cap → large-tx monitoring
    new_threshold = 15000.0
    new_desc = (
        "Large-transaction AML monitoring threshold (TND). "
        "NOTE: The TND 5,000 cash-payment cap was repealed by Finance Law 2026. "
        "This threshold is now a general enhanced-due-diligence trigger, not a structuring cap."
    )
    cursor.execute(
        """
        UPDATE risk_rules
        SET threshold = ?, description = ?, updated_at = CURRENT_TIMESTAMP
        WHERE rule_name = 'high_value'
        """,
        (new_threshold, new_desc),
    )

    cursor.execute(
        """
        INSERT INTO rule_change_log
            (rule_table, rule_name, change_type, old_value, new_value, changed_by, reason)
        VALUES ('risk_rules', 'high_value', 'UPDATE',
                ?, ?, ?, ?)
        """,
        (
            f"threshold={old_threshold}, description={old_desc!r}",
            f"threshold={new_threshold}, description={new_desc!r}",
            _CHANGED_BY,
            _REASON,
        ),
    )

    # 3. Add velocity-based smurfing parameters as risk_rules entries
    smurfing_rules = [
        (
            "smurfing_velocity_unit_cap",
            "smurfing",
            0.0,
            3000.0,
            1,
            "Per-transaction amount ceiling (TND) below which a tx qualifies as a smurfing unit. "
            "Multiple transactions below this cap accumulating above smurfing_velocity_agg_min "
            "are flagged as velocity smurfing. Replaces hard cash-cap dependency.",
        ),
        (
            "smurfing_velocity_agg_min",
            "smurfing",
            0.0,
            9000.0,
            1,
            "Minimum window-aggregate amount (TND) required to trigger the smurfing_velocity rule. "
            "i.e., v_count * avg_amount must exceed this value.",
        ),
        (
            "smurfing_velocity_min_count",
            "smurfing",
            0.0,
            3.0,
            1,
            "Minimum number of qualifying transactions in the 5-min window to trigger smurfing_velocity.",
        ),
    ]

    for rule_name, rule_type, weight, threshold, is_active, description in smurfing_rules:
        cursor.execute(
            """
            INSERT OR IGNORE INTO risk_rules
                (rule_name, rule_type, weight, threshold, is_active, description, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (rule_name, rule_type, weight, threshold, is_active, description, _CHANGED_BY),
        )
        # Log only newly inserted rows
        if cursor.rowcount > 0:
            cursor.execute(
                """
                INSERT INTO rule_change_log
                    (rule_table, rule_name, change_type, old_value, new_value, changed_by, reason)
                VALUES ('risk_rules', ?, 'INSERT', 'none', ?, ?, ?)
                """,
                (
                    rule_name,
                    f"threshold={threshold}",
                    _CHANGED_BY,
                    _REASON,
                ),
            )

    # 4. Record migration
    cursor.execute(
        """
        INSERT OR IGNORE INTO migration_history (migration_id, description)
        VALUES (?, ?)
        """,
        (
            _MIGRATION_ID,
            "Finance Law 2026: high_value threshold 5000→15000, velocity smurfing rules added",
        ),
    )

    conn.commit()
    conn.close()
    print(
        "Migration 0004 applied: high_value threshold updated to 15,000 TND; "
        "velocity-based smurfing rules inserted."
    )


def downgrade(db_path: str = None):
    if db_path is None:
        db_path = Path(__file__).parent.parent / "data" / "feedback.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE risk_rules
        SET threshold = 5000.0,
            description = 'High value transaction threshold (TND)',
            updated_at = CURRENT_TIMESTAMP
        WHERE rule_name = 'high_value'
        """
    )

    for rule_name in (
        "smurfing_velocity_unit_cap",
        "smurfing_velocity_agg_min",
        "smurfing_velocity_min_count",
    ):
        cursor.execute("DELETE FROM risk_rules WHERE rule_name = ?", (rule_name,))

    cursor.execute(
        "DELETE FROM migration_history WHERE migration_id = ?", (_MIGRATION_ID,)
    )

    conn.commit()
    conn.close()
    print("Migration 0004 downgraded: high_value threshold restored to 5,000 TND.")


if __name__ == "__main__":
    upgrade()
