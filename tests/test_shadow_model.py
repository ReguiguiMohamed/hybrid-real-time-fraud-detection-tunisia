"""Tests for shadow comparison tracking using shared SQLAlchemy database."""

import joblib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ml.shadow_model import ShadowModelManager
from shared.database import Base, ShadowModelRegistry, ShadowScoreLog


def _memory_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    return TestSession()


def test_registering_shadow_replaces_previous_model(tmp_path):
    session = _memory_session()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    joblib.dump({"name": "first"}, first / "pipeline.pkl")
    joblib.dump({"name": "second"}, second / "pipeline.pkl")
    manager = ShadowModelManager(db_session=session)

    assert manager.register_shadow_model(str(first), version_id="shadow-v1")
    assert manager.register_shadow_model(str(second), version_id="shadow-v2")

    assert manager.get_shadow_status()["version_id"] == "shadow-v2"
    statuses = {r.version_id: r.status for r in session.query(ShadowModelRegistry).all()}
    assert statuses == {"shadow-v1": "inactive", "shadow-v2": "shadow"}


def test_record_comparison_initializes_schema(tmp_path):
    session = _memory_session()
    manager = ShadowModelManager(db_session=session)

    manager.record_shadow_comparison("TXN_1", 0.8, 0.75)
    result = manager.compare_performance()

    assert result["comparisons"] == 1
    assert result["status"] == "comparing"


def test_alignment_does_not_claim_superiority(tmp_path):
    session = _memory_session()
    manager = ShadowModelManager(db_session=session)
    for index in range(10):
        manager.record_shadow_comparison(f"TXN_{index}", 0.8, 0.79)

    result = manager.compare_performance()

    assert result["recommendation"].startswith("ALIGNED")
    assert "labeled evaluation" in result["recommendation"]


def test_large_disagreement_requires_review(tmp_path):
    session = _memory_session()
    manager = ShadowModelManager(db_session=session)
    manager.record_shadow_comparison("TXN_1", 0.9, 0.1)
    manager.record_shadow_comparison("TXN_2", 0.1, 0.9)

    result = manager.compare_performance()

    assert result["recommendation"].startswith("HOLD")


def test_label_join_analysis(tmp_path):
    session = _memory_session()
    manager = ShadowModelManager(db_session=session)

    manager.record_shadow_comparison("TXN_1", 0.95, 0.92, champion_label=1, shadow_label=1, analyst_label="1")
    manager.record_shadow_comparison("TXN_2", 0.8, 0.3, champion_label=1, shadow_label=0, analyst_label="0")

    result = manager.compare_performance()

    assert result["label_analysis"] is not None
    assert result["label_analysis"]["labelled_count"] == 2
    assert result["label_analysis"]["champion_correct_count"] == 1
    assert result["label_analysis"]["shadow_correct_count"] == 2


def test_latency_recording(tmp_path):
    session = _memory_session()
    manager = ShadowModelManager(db_session=session)

    manager.record_shadow_comparison("TXN_1", 0.8, 0.75, latency_ms=12.5)
    manager.record_shadow_comparison("TXN_2", 0.7, 0.68, latency_ms=8.3)

    result = manager.compare_performance()

    assert result["latency"] is not None
    assert result["latency"]["avg_ms"] == 10.4
    assert result["latency"]["p95_ms"] is not None


def test_batch_comparison(tmp_path):
    session = _memory_session()
    manager = ShadowModelManager(db_session=session)

    manager.record_batch_comparison(
        [
            {"transaction_id": "TXN_1", "champion_score": 0.9, "shadow_score": 0.85},
            {"transaction_id": "TXN_2", "champion_score": 0.1, "shadow_score": 0.15},
        ]
    )

    result = manager.compare_performance()

    assert result["comparisons"] == 2
    assert result["status"] == "comparing"


def test_shadow_status_updates_registry(tmp_path):
    session = _memory_session()
    first = tmp_path / "shadow"
    first.mkdir()
    joblib.dump({"name": "shadow"}, first / "pipeline.pkl")
    manager = ShadowModelManager(db_session=session)

    assert manager.register_shadow_model(str(first), version_id="shadow-v1")
    assert manager.get_shadow_status()["active"] is True
    assert manager.get_shadow_status()["version_id"] == "shadow-v1"

    manager.record_shadow_comparison("TXN_1", 0.8, 0.75)
    manager.record_shadow_comparison("TXN_2", 0.7, 0.68)
    result = manager.compare_performance()

    assert result["comparisons"] == 2

    status = manager.get_shadow_status()
    assert status["total_comparisons"] == 2
