import json

from compliance.pkyc import PKYCPublisher, PKYCTriggerEvaluator
from shared.pii_masking import hash_pii


class CapturingProducer:
    def __init__(self):
        self.messages = []

    def produce(self, topic, key=None, value=None):
        self.messages.append({"topic": topic, "key": key, "value": value})

    def poll(self, timeout):
        return None


def test_low_risk_to_high_score_trigger_masks_account():
    tx = {
        "transaction_id": "TXN_PKYC_001",
        "user_id": "USER_123",
        "previous_risk_tier": "LOW",
    }

    event = PKYCTriggerEvaluator.evaluate(tx, 0.82)

    assert event is not None
    assert event.event_type == "pKYC_trigger"
    assert event.account_id == hash_pii("USER_123")
    assert event.account_id != "USER_123"
    assert event.trigger_reason == "LOW_RISK_TO_HIGH_SCORE"
    assert event.current_risk_tier == "HIGH"


def test_fcy_open_velocity_and_ring_reasons_are_combined():
    tx = {
        "transaction_id": "TXN_PKYC_002",
        "user_id": "USER_456",
        "account_type": "FCY",
        "account_event_type": "FCY_ACCOUNT_OPENED",
        "fcy_currency": "EUR",
        "current_week_tx_count": 25,
        "previous_week_tx_count": 5,
        "transaction_ring_detected": True,
    }

    event = PKYCTriggerEvaluator.evaluate(tx, 0.66)

    assert event is not None
    reasons = set(event.trigger_reason.split("|"))
    assert reasons == {
        "FCY_ACCOUNT_OPENED",
        "VELOCITY_WOW_SPIKE",
        "TRANSACTION_RING_DETECTED",
    }
    assert event.signals["velocity_week_over_week_pct"] == 400.0
    assert event.signals["fcy_currency"] == "EUR"


def test_no_trigger_without_explicit_trigger_signal():
    tx = {
        "transaction_id": "TXN_PKYC_003",
        "user_id": "USER_789",
        "account_type": "TND",
    }

    assert PKYCTriggerEvaluator.evaluate(tx, 0.69) is None


def test_publisher_serializes_event_to_configured_topic():
    producer = CapturingProducer()
    publisher = PKYCPublisher(
        bootstrap_servers="unused:9092",
        topic="pkyc_test_topic",
        producer=producer,
        audit_db_path=":memory:",
    )
    tx = {
        "transaction_id": "TXN_PKYC_004",
        "user_id": "USER_999",
        "previous_ml_probability": 0.1,
    }

    event = publisher.publish_for_transaction(tx, 0.75)

    assert event is not None
    assert len(producer.messages) == 1
    message = producer.messages[0]
    assert message["topic"] == "pkyc_test_topic"
    assert message["key"] == hash_pii("USER_999").encode("utf-8")
    payload = json.loads(message["value"].decode("utf-8"))
    assert payload["event_type"] == "pKYC_trigger"
    assert payload["transaction_id"] == "TXN_PKYC_004"


def test_publisher_records_auditable_trigger_without_raw_account(tmp_path):
    import sqlite3

    db_path = tmp_path / "pkyc_audit.db"
    publisher = PKYCPublisher(
        bootstrap_servers="unused:9092",
        topic="pkyc_test_topic",
        producer=None,
        audit_db_path=str(db_path),
    )
    tx = {
        "transaction_id": "TXN_PKYC_005",
        "user_id": "USER_AUDIT",
        "previous_risk_tier": "LOW",
    }

    event = publisher.publish_for_transaction(tx, 0.78)

    assert event is not None
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_type, account_id, trigger_reason, current_risk_tier, transaction_id
        FROM pkyc_triggers
    """)
    row = cursor.fetchone()
    conn.close()

    assert row == (
        "pKYC_trigger",
        hash_pii("USER_AUDIT"),
        "LOW_RISK_TO_HIGH_SCORE",
        "HIGH",
        "TXN_PKYC_005",
    )
    assert row[1] != "USER_AUDIT"
