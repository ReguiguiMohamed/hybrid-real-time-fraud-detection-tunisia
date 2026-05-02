import json

from compliance.pkyc import PKYCTriggerEvaluator, PKYCPublisher
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
