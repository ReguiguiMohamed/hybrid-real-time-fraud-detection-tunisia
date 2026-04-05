"""Tests for the ForensicAnalyticEngine monitoring module."""
import pytest
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from monitoring import ForensicAnalyticEngine


class TestPerformanceMetrics:
    def test_empty_latencies(self, tmp_db):
        engine = ForensicAnalyticEngine(db_path=tmp_db)
        metrics = engine.get_performance_metrics()
        assert metrics["total_calls"] == 0
        assert metrics["avg_latency_ms"] == 0

    def test_record_and_get_latencies(self, tmp_db):
        engine = ForensicAnalyticEngine(db_path=tmp_db)
        for ms in [10.0, 20.0, 30.0, 40.0, 50.0]:
            engine.record_inference_latency(ms)

        metrics = engine.get_performance_metrics()
        assert metrics["total_calls"] == 5
        assert metrics["avg_latency_ms"] == 30.0


class TestFeedbackAnalysis:
    def test_empty_database(self, tmp_db):
        engine = ForensicAnalyticEngine(db_path=tmp_db)
        analysis = engine.get_feedback_analysis()
        assert analysis["total_feedback"] == 0
        assert analysis["precision"] == 0

    def test_with_populated_data(self, populated_db):
        engine = ForensicAnalyticEngine(db_path=populated_db)
        analysis = engine.get_feedback_analysis()
        assert analysis["total_feedback"] == 3
        assert "Confirmed Fraud" in analysis["feedback_counts"]
        assert analysis["precision"] > 0


class TestThresholdAnalysis:
    def test_empty_returns_default(self, tmp_db):
        engine = ForensicAnalyticEngine(db_path=tmp_db)
        result = engine.get_ml_threshold_analysis()
        assert result["optimal_threshold"] == 0.85


class TestDistributionComparison:
    def test_disallowed_column(self, populated_db):
        engine = ForensicAnalyticEngine(db_path=populated_db)
        result = engine.get_distribution_comparison("user_id")  # Not in whitelist
        assert "error" in result

    def test_allowed_column(self, populated_db):
        engine = ForensicAnalyticEngine(db_path=populated_db)
        result = engine.get_distribution_comparison("amount_tnd")
        assert "current_values" in result
        assert "baseline_values" in result
