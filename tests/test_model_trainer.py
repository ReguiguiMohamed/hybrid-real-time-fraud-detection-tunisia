"""Tests for retraining policy without starting Spark."""

from ml.train_model import FraudModelTrainer


class EmptyRepository:
    def ensure_schema(self):
        return None

    def count_verified_feedback(self):
        return 0


def test_insufficient_feedback_skips_without_starting_spark(monkeypatch):
    monkeypatch.setenv("RETRAIN_MIN_FEEDBACK", "10")
    monkeypatch.setenv("RETRAIN_ON_DRIFT_ENABLED", "false")
    trainer = FraudModelTrainer(repository=EmptyRepository())

    assert trainer.train_champion_challenger() is False
    assert trainer._spark is None


def test_metric_improvement_requires_human_approval():
    decision = FraudModelTrainer._promotion_decision(
        {"f1_score": 0.85},
        {"f1_score": 0.80},
        threshold=0.02,
        approved_by=None,
    )

    assert decision["metric_eligible"] is True
    assert decision["approved"] is False
    assert decision["promote"] is False


def test_human_approval_does_not_override_metric_gate():
    decision = FraudModelTrainer._promotion_decision(
        {"f1_score": 0.79},
        {"f1_score": 0.80},
        threshold=0.02,
        approved_by="risk-officer",
    )

    assert decision["metric_eligible"] is False
    assert decision["promote"] is False


def test_first_champion_requires_human_approval():
    decision = FraudModelTrainer._promotion_decision(
        {"f1_score": 0.75},
        None,
        threshold=0.02,
        approved_by="risk-officer",
    )

    assert decision["metric_eligible"] is True
    assert decision["promote"] is True
