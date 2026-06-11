"""Tests for model training status reads."""

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ml.model_repository import ModelRepository
from shared.database import Base, ModelRegistry


def build_repository(tmp_path):
    bind = create_engine(
        f"sqlite:///{(tmp_path / 'models.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=bind)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=bind)
    return ModelRepository(factory), factory


def test_empty_registry_has_empty_status(tmp_path):
    repository, _ = build_repository(tmp_path)

    assert repository.get_champion_training_status() == {
        "last_success_at": None,
        "last_failure_at": None,
        "last_error": None,
    }


def test_champion_success_is_returned(tmp_path):
    repository, factory = build_repository(tmp_path)
    trained_at = datetime(2026, 6, 1, 10, 30)
    with factory() as db:
        db.add(
            ModelRegistry(
                version_id="v1",
                model_path="models/v1",
                is_champion=1,
                promoted_at=trained_at,
                last_training_success_at=trained_at,
            )
        )
        db.commit()

    status = repository.get_champion_training_status()

    assert status["last_success_at"] == "2026-06-01T10:30:00"
    assert status["last_error"] is None


def test_champion_failure_is_returned(tmp_path):
    repository, factory = build_repository(tmp_path)
    failed_at = datetime(2026, 6, 2, 8, 15)
    with factory() as db:
        db.add(
            ModelRegistry(
                version_id="v2",
                model_path="models/v2",
                is_champion=1,
                promoted_at=failed_at,
                last_training_failure_at=failed_at,
                last_training_error="Training failed",
            )
        )
        db.commit()

    status = repository.get_champion_training_status()

    assert status["last_failure_at"] == "2026-06-02T08:15:00"
    assert status["last_error"] == "Training failed"
