"""Tests for the SQLAlchemy model lifecycle persistence boundary."""

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ml.model_repository import ModelRepository
from shared.database import AuditLog, Base, FeedbackLabel, HighRiskAlert, ModelRegistry


def build_repository(tmp_path):
    bind = create_engine(
        f"sqlite:///{(tmp_path / 'models.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
    repository = ModelRepository(factory, bind)
    repository.ensure_schema()
    return repository, factory


def test_reads_verified_feedback_from_shared_schema(tmp_path):
    repository, factory = build_repository(tmp_path)
    with factory() as db:
        db.add(
            HighRiskAlert(
                transaction_id="TXN_REPO_001",
                user_id="USER_1",
                amount_tnd=1200.0,
                governorate="Tunis",
                payment_method="card",
                ml_probability=0.9,
            )
        )
        db.add(FeedbackLabel(transaction_id="TXN_REPO_001", analyst_label="Confirmed Fraud"))
        db.commit()

    rows = repository.get_verified_feedback_rows()

    assert repository.count_verified_feedback() == 1
    assert rows[0][-1] == 1


def test_promoting_model_demotes_previous_champion(tmp_path):
    repository, factory = build_repository(tmp_path)
    promoted_at = datetime.now(timezone.utc).replace(tzinfo=None)
    repository.record_model(
        version_id="v1",
        model_path="models/v1",
        f1_score=0.7,
        auc=0.8,
        is_champion=True,
        promoted_at=promoted_at,
        training_samples_count=100,
        feature_importance="[]",
    )
    repository.record_model(
        version_id="v2",
        model_path="models/v2",
        f1_score=0.8,
        auc=0.9,
        is_champion=True,
        promoted_at=promoted_at,
        training_samples_count=120,
        feature_importance="[]",
    )

    with factory() as db:
        champions = db.query(ModelRegistry).filter(ModelRegistry.is_champion == 1).all()

    assert [model.version_id for model in champions] == ["v2"]
    assert repository.get_current_champion()["version_id"] == "v2"


def test_audit_event_is_persisted(tmp_path):
    repository, factory = build_repository(tmp_path)

    repository.log_audit_event(
        entity_type="MODEL",
        entity_id="v2",
        action="PROMOTE",
        user_id="risk-officer",
        previous_state=None,
        new_state='{"version_id": "v2"}',
    )

    with factory() as db:
        event = db.query(AuditLog).one()

    assert event.entity_id == "v2"
    assert event.action == "PROMOTE"
