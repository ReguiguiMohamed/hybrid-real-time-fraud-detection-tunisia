"""
Prometheus Metrics Exporter for Amastan Fraud Shield Guard
Exposes pipeline metrics at /metrics endpoint for Prometheus scraping.
"""
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import time
import threading
from collections import deque
from typing import Optional


# Registry
registry = CollectorRegistry()

# ==================== Pipeline Metrics ====================

# Throughput
fraud_predictions_total = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions made",
    ["alert_type"],
    registry=registry,
)

fraud_alerts_total = Counter(
    "fraud_alerts_total",
    "Total number of alerts generated",
    ["alert_type", "status"],  # status: success, failed
    registry=registry,
)

kafka_messages_consumed = Counter(
    "kafka_messages_consumed_total",
    "Total Kafka messages consumed",
    registry=registry,
)

kafka_messages_failed = Counter(
    "kafka_messages_failed_total",
    "Total Kafka messages that failed to process",
    ["error_type"],
    registry=registry,
)

# Latency
fraud_ingestion_latency = Histogram(
    "fraud_ingestion_latency_seconds",
    "Time from event timestamp to processing completion",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=registry,
)

ml_inference_latency = Histogram(
    "ml_inference_latency_seconds",
    "Time taken for ML model inference per batch",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=registry,
)

api_call_latency = Histogram(
    "api_call_latency_seconds",
    "Time taken for API calls (alert submission)",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry,
)

sar_generation_latency = Histogram(
    "sar_generation_latency_seconds",
    "Time taken for SAR report generation via RAG",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=registry,
)

# Gauges (current state)
dlq_pending_count = Gauge(
    "dlq_pending_count",
    "Number of pending items in dead letter queue",
    registry=registry,
)

dlq_failed_count = Gauge(
    "dlq_failed_count",
    "Number of permanently failed items in DLQ",
    registry=registry,
)

dlq_success_count = Gauge(
    "dlq_success_count",
    "Number of successfully retried items from DLQ",
    registry=registry,
)

model_f1_score = Gauge(
    "model_f1_score",
    "Current champion model F1 score",
    registry=registry,
)

model_auc = Gauge(
    "model_auc",
    "Current champion model AUC score",
    registry=registry,
)

model_version = Gauge(
    "model_version_info",
    "Current champion model version (value is timestamp)",
    registry=registry,
)

# Feedback metrics
feedback_confirmed_fraud_total = Counter(
    "feedback_confirmed_fraud_total",
    "Total confirmed fraud by analysts",
    registry=registry,
)

feedback_false_positive_total = Counter(
    "feedback_false_positive_total",
    "Total false positives identified by analysts",
    registry=registry,
)

feedback_precision = Gauge(
    "feedback_precision",
    "Current precision rate based on analyst feedback",
    registry=registry,
)

feedback_pending_count = Gauge(
    "feedback_pending_count",
    "Number of alerts awaiting analyst review",
    registry=registry,
)

# Consumer lag (reported by consumer)
kafka_consumer_lag = Gauge(
    "kafka_consumer_lag",
    "Kafka consumer group lag in messages",
    ["consumer_group", "topic", "partition"],
    registry=registry,
)

# Spark metrics
spark_streaming_batch_processing_time = Gauge(
    "spark_streaming_batch_processing_time_seconds",
    "Current Spark batch processing time",
    registry=registry,
)

spark_active_batches = Gauge(
    "spark_active_batches",
    "Number of currently active Spark micro-batches",
    registry=registry,
)


# ==================== Metrics Helper ====================

class MetricsCollector:
    """Thread-safe metrics collector for pipeline components."""

    def __init__(self):
        self._latencies = deque(maxlen=10000)
        self._lock = threading.Lock()

    def record_prediction(self, alert_type: str = "unknown"):
        fraud_predictions_total.labels(alert_type=alert_type).inc()

    def record_alert(self, alert_type: str = "unknown", status: str = "success"):
        fraud_alerts_total.labels(alert_type=alert_type, status=status).inc()

    def record_ingestion_latency(self, seconds: float):
        fraud_ingestion_latency.observe(seconds)
        with self._lock:
            self._latencies.append(seconds)

    def record_ml_latency(self, seconds: float):
        ml_inference_latency.observe(seconds)

    def record_api_latency(self, seconds: float):
        api_call_latency.observe(seconds)

    def record_sar_latency(self, seconds: float):
        sar_generation_latency.observe(seconds)

    def record_kafka_consumed(self):
        kafka_messages_consumed.inc()

    def record_kafka_failure(self, error_type: str = "unknown"):
        kafka_messages_failed.labels(error_type=error_type).inc()

    def update_dlq_metrics(self, pending: int = 0, failed: int = 0, success: int = 0):
        dlq_pending_count.set(pending)
        dlq_failed_count.set(failed)
        dlq_success_count.set(success)

    def update_model_metrics(self, f1: float, auc: float, version_timestamp: float):
        model_f1_score.set(f1)
        model_auc.set(auc)
        model_version.set(version_timestamp)

    def update_feedback_metrics(
        self,
        confirmed: int = 0,
        false_positives: int = 0,
        precision: float = 0.0,
        pending: int = 0,
    ):
        feedback_confirmed_fraud_total.inc(confirmed)
        feedback_false_positive_total.inc(false_positives)
        feedback_precision.set(precision)
        feedback_pending_count.set(pending)

    def update_consumer_lag(self, lag: int, consumer_group: str = "fraud-consumer", topic: str = "tunisian_transactions", partition: int = 0):
        kafka_consumer_lag.labels(
            consumer_group=consumer_group, topic=topic, partition=partition
        ).set(lag)

    def update_spark_metrics(self, processing_time: float, active_batches: int = 1):
        spark_streaming_batch_processing_time.set(processing_time)
        spark_active_batches.set(active_batches)

    def get_p95_latency(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            sorted_latencies = sorted(self._latencies)
            idx = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def get_p99_latency(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            sorted_latencies = sorted(self._latencies)
            idx = int(len(sorted_latencies) * 0.99)
            return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


# Singleton instance
metrics_collector = MetricsCollector()


def metrics_endpoint():
    """FastAPI endpoint function to expose Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}
