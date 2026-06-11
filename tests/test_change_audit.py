import json

from compliance.change_audit import append_change_audit_event


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
