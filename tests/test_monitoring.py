"""Tests for the ForensicAnalyticEngine monitoring module."""

import json

from monitoring import ForensicAnalyticEngine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def build_engine(db_path):
    bind = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    return ForensicAnalyticEngine(sessionmaker(bind=bind))


class TestPerformanceMetrics:
    def test_empty_latencies(self, tmp_db):
        engine = build_engine(tmp_db)
        metrics = engine.get_performance_metrics()
        assert metrics["total_calls"] == 0
        assert metrics["avg_latency_ms"] == 0

    def test_record_and_get_latencies(self, tmp_db):
        engine = build_engine(tmp_db)
        for ms in [10.0, 20.0, 30.0, 40.0, 50.0]:
            engine.record_inference_latency(ms)

        metrics = engine.get_performance_metrics()
        assert metrics["total_calls"] == 5
        assert metrics["avg_latency_ms"] == 30.0


class TestFeedbackAnalysis:
    def test_empty_database(self, tmp_db):
        engine = build_engine(tmp_db)
        analysis = engine.get_feedback_analysis()
        assert analysis["total_feedback"] == 0
        assert analysis["precision"] == 0

    def test_with_populated_data(self, populated_db):
        engine = build_engine(populated_db)
        analysis = engine.get_feedback_analysis()
        assert analysis["total_feedback"] == 3
        assert "Confirmed Fraud" in analysis["feedback_counts"]
        assert analysis["precision"] > 0


class TestThresholdAnalysis:
    def test_empty_returns_default(self, tmp_db):
        engine = build_engine(tmp_db)
        result = engine.get_ml_threshold_analysis()
        assert result["optimal_threshold"] == 0.85


class TestDistributionComparison:
    def test_disallowed_column(self, populated_db):
        engine = build_engine(populated_db)
        result = engine.get_distribution_comparison("user_id")  # Not in whitelist
        assert "error" in result

    def test_allowed_column(self, populated_db):
        engine = build_engine(populated_db)
        result = engine.get_distribution_comparison("amount_tnd")
        assert "current_values" in result
        assert "baseline_values" in result


class TestDriftRetraining:
    def test_psi_detects_material_feature_drift(self, tmp_db):
        engine = build_engine(tmp_db)

        result = engine.detect_feature_drift_psi(
            "avg_amount",
            current_values=[9000, 9500, 10000, 11000, 12000],
            reference_values=[100, 120, 140, 160, 180],
            threshold=0.2,
        )

        assert result["feature"] == "avg_amount"
        assert result["psi"] > 0.2
        assert result["trigger_retraining"] is True

    def test_psi_ignores_stable_distribution(self, tmp_db):
        engine = build_engine(tmp_db)

        result = engine.detect_feature_drift_psi(
            "v_count",
            current_values=[1, 2, 3, 4, 5],
            reference_values=[1, 2, 3, 4, 5],
            threshold=0.2,
        )

        assert result["psi"] == 0.0
        assert result["trigger_retraining"] is False

    def test_page_hinkley_detects_score_shift(self, tmp_db):
        engine = build_engine(tmp_db)
        scores = [0.05] * 20 + [0.75] * 20

        result = engine.page_hinkley_test(scores, delta=0.005, lambda_=0.05)

        assert result["drift_detected"] is True
        assert result["max_statistic"] > 0.05

    def test_drift_trigger_writes_audit_event(self, tmp_db, tmp_path):
        engine = build_engine(tmp_db)
        audit_path = tmp_path / "drift_audit.jsonl"
        psi_result = engine.detect_feature_drift_psi(
            "avg_amount",
            current_values=[9000, 9500, 10000, 11000, 12000],
            reference_values=[100, 120, 140, 160, 180],
            threshold=0.2,
        )

        decision = engine.evaluate_drift_retraining_trigger(
            psi_results=[psi_result],
            score_drift_result={"drift_detected": False},
            audit_log_path=str(audit_path),
            actor="monitoring-test",
        )

        assert decision["trigger_retraining"] is True
        assert decision["trigger_reason"] == "PSI>0.2:avg_amount"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["event_type"] == "RETRAINING_TRIGGER"
        assert audit["actor"] == "monitoring-test"
        assert audit["trigger_reason"] == "PSI>0.2:avg_amount"
        assert len(audit["entry_hash"]) == 64

    def test_drift_assessment_returns_decision_shape(self, populated_db):
        engine = build_engine(populated_db)

        assessment = engine.get_drift_retraining_assessment()

        assert "psi_results" in assessment
        assert "score_drift" in assessment
        assert "decision" in assessment
        assert assessment["basis"] == "Recorded alert feature distributions and model score stream."
