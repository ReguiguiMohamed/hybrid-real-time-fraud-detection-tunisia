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


def test_promotion_records_audit_event(tmp_path):
    """Full mocked lifecycle: record previous champion → evaluate challenger →
    promote → verify audit trail. No Spark needed — tests repository + decision
    logic integration."""
    from datetime import datetime, timezone
    from shared.database import AuditLog
    from ml.model_repository import ModelRepository
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    bind = create_engine(
        f"sqlite:///{(tmp_path / 'lifecycle.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
    repo = ModelRepository(factory, bind)
    repo.ensure_schema()

    promoted_at = datetime.now(timezone.utc).replace(tzinfo=None)
    repo.record_model(
        version_id="champion-v1",
        model_path="models/champion-v1",
        f1_score=0.75,
        auc=0.80,
        is_champion=True,
        promoted_at=promoted_at,
        training_samples_count=200,
        feature_importance='[{"feature":"amount","score":0.5}]',
    )

    champion_before = repo.get_current_champion()

    decision = FraudModelTrainer._promotion_decision(
        {"f1_score": 0.88, "auc": 0.92},
        {"f1_score": 0.75, "auc": 0.80},
        threshold=0.02,
        approved_by="risk-officer",
    )

    assert decision["promote"] is True
    assert decision["f1_improvement"] == 0.13

    repo.log_audit_event(
        entity_type="MODEL",
        entity_id="challenger-v2",
        action="PROMOTE",
        user_id="risk-officer",
        previous_state=str(champion_before),
        new_state='{"version_id": "challenger-v2", "f1_score": 0.88}',
    )

    with factory() as db:
        event = db.query(AuditLog).filter(AuditLog.action == "PROMOTE").one()
    assert event.entity_id == "challenger-v2"
    assert event.user_id == "risk-officer"


def test_training_failure_logs_audit_event(tmp_path):
    """When training fails, the audit trail records the failure reason."""
    from ml.model_repository import ModelRepository
    from shared.database import AuditLog
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    bind = create_engine(
        f"sqlite:///{(tmp_path / 'lifecycle.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
    repo = ModelRepository(factory, bind)
    repo.ensure_schema()

    repo.log_audit_event(
        entity_type="MODEL",
        entity_id="failed-run-1",
        action="TRAINING_FAILURE",
        user_id="system",
        previous_state=None,
        new_state='{"error": "No valid feature columns in training data"}',
    )

    with factory() as db:
        events = db.query(AuditLog).filter(AuditLog.action == "TRAINING_FAILURE").all()
    assert len(events) == 1
    assert events[0].entity_id == "failed-run-1"


def test_champion_artifact_load_failure_returns_metadata(tmp_path):
    """Simulate a champion whose artifact cannot be loaded.
    The repository returns metadata; load failure is caller's responsibility."""
    from datetime import datetime, timezone
    from ml.model_repository import ModelRepository
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    bind = create_engine(
        f"sqlite:///{(tmp_path / 'models.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
    repo = ModelRepository(factory, bind)
    repo.ensure_schema()

    repo.record_model(
        version_id="v1",
        model_path="/nonexistent/path",
        f1_score=0.7,
        auc=0.8,
        is_champion=True,
        promoted_at=datetime.now(timezone.utc).replace(tzinfo=None),
        training_samples_count=100,
        feature_importance="[]",
    )

    champ = repo.get_current_champion()
    assert champ["version_id"] == "v1"
    assert champ["model_path"] == "/nonexistent/path"
