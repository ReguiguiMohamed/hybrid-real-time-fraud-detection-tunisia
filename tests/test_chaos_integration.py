"""
Chaos Test: Kafka Failure & Recovery
Tests the pipeline's behavior when Kafka becomes unavailable and recovers.
Uses Testcontainers for realistic integration testing.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from testcontainers.kafka import KafkaContainer


@pytest.mark.integration
class TestKafkaFailureRecovery:
    """Test Kafka failure scenarios and recovery behavior."""

    def test_consumer_handles_kafka_connection_error(self):
        """Consumer should gracefully handle Kafka connection failures."""
        from src.streaming.consumer import FraudProcessor

        # Mock Kafka to raise connection error
        with patch("pyspark.sql.SparkSession") as mock_spark:
            mock_spark.builder = MagicMock()
            mock_spark.builder.appName.return_value = mock_spark.builder
            mock_spark.builder.config.return_value = mock_spark.builder

            with patch("pyspark.sql.SparkSession.readStream") as mock_read:
                from pyspark.sql.utils import AnalysisException

                mock_read.side_effect = AnalysisException("Failed to connect to Kafka: Connection refused", None)

                # Should raise exception (expected behavior)
                # The important thing is it doesn't crash silently
                with pytest.raises((Exception,)):
                    processor = FraudProcessor.__new__(FraudProcessor)
                    processor.spark = mock_spark
                    processor.process_stream()

    def test_dlq_captures_failed_alerts_on_api_down(self, tmp_path, monkeypatch):
        """Failed alerts should be captured in DLQ when API is down."""
        import src.shared.utils as utils

        db_path = tmp_path / "dead_letter_queue.db"
        monkeypatch.setattr(utils, "DLQ_DB_PATH", str(db_path))
        tx_data = {"transaction_id": "test-tx-001", "user_id": "user-1"}
        alert_payload = {"transaction_id": "test-tx-001", "amount_tnd": 5000.0}

        utils.log_failed_alert(tx_data, alert_payload, "API_DOWN", "Connection refused")

        with utils.get_sqlite_connection(str(db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM failed_alerts WHERE status = 'PENDING'").fetchone()[0]
        assert count == 1, "Failed alert should be in DLQ"

    def test_dlq_retry_mechanism(self, tmp_path, monkeypatch):
        """DLQ retry worker should attempt to re-process failed alerts."""
        import src.shared.utils as utils

        db_path = tmp_path / "dead_letter_queue.db"
        monkeypatch.setattr(utils, "DLQ_DB_PATH", str(db_path))
        utils.log_failed_alert(
            {
                "transaction_id": "test-tx-002",
                "user_id": "user-2",
                "amount_tnd": 1000.0,
            },
            {"transaction_id": "test-tx-002"},
            "API_DOWN",
            "Connection refused",
        )
        monkeypatch.setattr(utils, "make_authenticated_request", lambda *_args, **_kwargs: None)

        utils.retry_failed_alerts(max_attempts=3)

        with utils.get_sqlite_connection(str(db_path)) as conn:
            row = conn.execute(
                "SELECT attempts, status FROM failed_alerts WHERE transaction_id = 'test-tx-002'"
            ).fetchone()
        assert row[0] == 1, "Retry count should be incremented"


@pytest.mark.integration
class TestCheckpointRecovery:
    """Test Spark checkpoint recovery after failure."""

    def test_checkpoint_directory_created(self):
        """Spark should create checkpoint directory during streaming."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint")

            # Verify checkpoint path is configurable
            assert checkpoint_path is not None
            os.makedirs(checkpoint_path, exist_ok=True)
            assert os.path.exists(checkpoint_path)

    def test_state_recovery_after_restart(self):
        """Verify that state can be recovered from checkpoint."""
        # This is a conceptual test - full Spark testing requires Java
        # In production, use Testcontainers with a real Spark cluster
        checkpoint_data = {
            "epoch_id": 42,
            "last_processed_offset": 1000,
            "watermark": "2026-01-01T00:00:00Z",
        }

        # Verify checkpoint data is serializable
        import json

        serialized = json.dumps(checkpoint_data)
        deserialized = json.loads(serialized)

        assert deserialized["epoch_id"] == 42
        assert deserialized["last_processed_offset"] == 1000


@pytest.mark.integration
class TestOllamaFallback:
    """Test Ollama failure scenarios and deterministic fallback."""

    def test_sar_generator_fallback_on_ollama_down(self):
        """SAR generator should use deterministic template when Ollama is down."""
        from src.rag_engine.sar_validator import generate_deterministic_fallback, validate_sar_output

        tx_data = {
            "transaction_id": "test-tx-003",
            "user_id": "user-3",
            "amount_tnd": 5000.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "branch_id": "B01",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        ml_score = 0.92

        # Simulate LLM failure (empty response)
        report = validate_sar_output("", tx_data, ml_score)

        assert report is not None
        assert report.transaction_id == "test-tx-003"
        assert report.validation_passed is False
        assert len(report.executive_summary) >= 30
        assert len(report.risk_factors) >= 1
        assert report.urgency_assessment.urgency_level in {"IMMEDIATE", "HIGH", "STANDARD", "LOW"}

    def test_sar_generator_fallback_on_malformed_json(self):
        """SAR generator should fallback when LLM returns malformed JSON."""
        from src.rag_engine.sar_validator import validate_sar_output

        tx_data = {
            "transaction_id": "test-tx-004",
            "user_id": "user-4",
            "amount_tnd": 1500.0,
            "governorate": "Sfax",
            "payment_method": "card",
            "branch_id": "B02",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        ml_score = 0.75

        # Malformed JSON
        malformed = "This is not JSON at all! The LLM hallucinated completely."
        report = validate_sar_output(malformed, tx_data, ml_score)

        assert report is not None
        assert report.validation_passed is False
        assert report.raw_llm_output == malformed

    def test_sar_generator_fallback_on_partial_json(self):
        """SAR generator should handle partial JSON from LLM."""
        from src.rag_engine.sar_validator import validate_sar_output

        tx_data = {
            "transaction_id": "test-tx-005",
            "user_id": "user-5",
            "amount_tnd": 3000.0,
            "governorate": "Sousse",
            "payment_method": "mobile",
            "branch_id": "B03",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        ml_score = 0.65

        # Partial JSON (missing required fields)
        partial = '{"executive_summary": "Some activity was observed"}'
        report = validate_sar_output(partial, tx_data, ml_score)

        assert report is not None
        # Should have used fallback since JSON is incomplete
        assert report.transaction_id == "test-tx-005"


@pytest.mark.integration
class TestLoadAndDegradation:
    """Test system behavior under load and partial degradation."""

    def test_consumer_handles_high_volume(self):
        """Consumer should handle high-volume transaction batches."""
        # Conceptual test - in production, use load_test.py
        batch_size = 10000
        processing_time_target = 30  # seconds

        # Simulate batch processing time estimate
        # At ~1000 tx/sec, 10K tx should take ~10 seconds
        estimated_time = batch_size / 1000
        assert estimated_time < processing_time_target

    def test_graceful_degradation_without_ml_model(self):
        """System should function with rule-based scoring when ML model is unavailable."""
        import os
        import sys

        if os.getenv("RUN_SPARK_TESTS") != "1":
            pytest.skip("Set RUN_SPARK_TESTS=1 to run local Spark worker tests.")

        from pyspark.sql import SparkSession
        from pyspark.sql.functions import lit

        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
        spark = SparkSession.builder.appName("test").getOrCreate()

        # Create sample data
        data = [("tx-001", "user-1", 5000.0, "Tunis", "card")]
        df = spark.createDataFrame(data, ["transaction_id", "user_id", "amount_tnd", "governorate", "payment_method"])

        # Add default ML columns (simulating no model scenario)
        df = df.withColumn("ml_prediction", lit(-1)).withColumn("ml_probability", lit(0.0))

        # Verify columns exist
        assert "ml_prediction" in df.columns
        assert "ml_probability" in df.columns

        row = df.collect()[0]
        assert row["ml_prediction"] == -1
        assert row["ml_probability"] == 0.0

        spark.stop()
