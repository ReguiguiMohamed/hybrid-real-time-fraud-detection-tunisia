import json
import sqlite3

import pytest

from compliance.change_audit import append_change_audit_event
from shared.rules_engine import RulesEngine


def test_change_audit_events_are_hash_chained(tmp_path):
    audit_path = tmp_path / "change_audit.jsonl"

    first = append_change_audit_event(
        {"event_type": "RULE_CHANGE", "entity_id": "high_value", "action": "UPDATE"},
        audit_log_path=str(audit_path),
    )
    second = append_change_audit_event(
        {"event_type": "MODEL_PROMOTION", "entity_id": "model_v2", "action": "PROMOTE"},
        audit_log_path=str(audit_path),
    )

    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 2
    assert rows[0]["entry_hash"] == first["entry_hash"]
    assert rows[1]["entry_hash"] == second["entry_hash"]
    assert rows[0]["previous_hash"] is None
    assert rows[1]["previous_hash"] == rows[0]["entry_hash"]
    assert len(rows[1]["entry_hash"]) == 64


def test_rule_updates_write_tamper_evident_audit_event(tmp_path, monkeypatch):
    db_path = tmp_path / "rules.db"
    audit_path = tmp_path / "change_audit.jsonl"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE risk_rules (
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
    """
    )
    cursor.execute(
        """
        CREATE TABLE rule_change_log (
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
    """
    )
    cursor.execute(
        """
        INSERT INTO risk_rules (rule_name, rule_type, weight, threshold, description)
        VALUES ('high_value', 'amount_threshold', 0.2, 15000.0, 'Enhanced monitoring threshold')
    """
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("CHANGE_AUDIT_LOG", str(audit_path))
    engine = RulesEngine(str(db_path))

    updated = engine.update_rule(
        "high_value",
        threshold=16000.0,
        changed_by="risk-officer-ahmed",
        reason="Annual threshold review",
        regulatory_reference="Finance Law 2026 cash-cap repeal",
    )

    assert updated is True
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["event_type"] == "RULE_CHANGE"
    assert audit["actor"] == "risk-officer-ahmed"
    assert audit["entity_id"] == "high_value"
    assert audit["previous_state"]["threshold"] == 15000.0
    assert audit["new_state"]["threshold"] == 16000.0
    assert audit["related_regulatory_reference"] == "Finance Law 2026 cash-cap repeal"


def test_train_pipeline_blocks_champion_registration_without_human_approver(tmp_path, monkeypatch):
    from src.ml.train_pipeline import FraudDetectionPipeline

    monkeypatch.delenv("MODEL_PROMOTION_APPROVED_BY", raising=False)
    monkeypatch.setenv("CHANGE_AUDIT_LOG", str(tmp_path / "change_audit.jsonl"))
    trainer = FraudDetectionPipeline()
    trainer.feedback_db_path = str(tmp_path / "feedback.db")
    trainer.metrics = {"f1": 0.8, "pr_auc": 0.7, "train_samples": 12}

    with pytest.raises(RuntimeError, match="MODEL_PROMOTION_APPROVED_BY"):
        trainer._register_in_registry("model_without_approval", str(tmp_path / "model"))
