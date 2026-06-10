"""Tests for shadow comparison tracking semantics."""

import sqlite3

import joblib

from ml.shadow_model import ShadowModelManager


def test_registering_shadow_replaces_previous_model(tmp_path):
    database_path = tmp_path / "shadow.db"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    joblib.dump({"name": "first"}, first / "pipeline.pkl")
    joblib.dump({"name": "second"}, second / "pipeline.pkl")
    manager = ShadowModelManager(str(database_path))

    assert manager.register_shadow_model(str(first), version_id="shadow-v1")
    assert manager.register_shadow_model(str(second), version_id="shadow-v2")

    assert manager.get_shadow_status()["version_id"] == "shadow-v2"
    with sqlite3.connect(database_path) as conn:
        statuses = dict(conn.execute("SELECT version_id, status FROM shadow_model_registry").fetchall())
    assert statuses == {"shadow-v1": "inactive", "shadow-v2": "shadow"}


def test_record_comparison_initializes_schema(tmp_path):
    manager = ShadowModelManager(str(tmp_path / "shadow.db"))

    manager.record_shadow_comparison("TXN_1", 0.8, 0.75)
    result = manager.compare_performance()

    assert result["comparisons"] == 1
    assert result["status"] == "comparing"


def test_alignment_does_not_claim_superiority(tmp_path):
    manager = ShadowModelManager(str(tmp_path / "shadow.db"))
    for index in range(10):
        manager.record_shadow_comparison(f"TXN_{index}", 0.8, 0.79)

    result = manager.compare_performance()

    assert result["recommendation"].startswith("ALIGNED")
    assert "labeled evaluation" in result["recommendation"]


def test_large_disagreement_requires_review(tmp_path):
    manager = ShadowModelManager(str(tmp_path / "shadow.db"))
    manager.record_shadow_comparison("TXN_1", 0.9, 0.1)
    manager.record_shadow_comparison("TXN_2", 0.1, 0.9)

    result = manager.compare_performance()

    assert result["recommendation"].startswith("HOLD")
