"""
OpenTelemetry Tracing for Amastan Fraud Shield Guard
Provides distributed tracing across the entire fraud detection pipeline.

Integrates with:
- Jaeger / Zipkin for trace visualization
- Prometheus for metrics correlation
- Alertmanager for trace-based alerting

Usage:
    from src.shared.tracing import tracer, start_span

    # In the consumer
    with start_span("process_batch") as span:
        span.set_attribute("batch_size", len(rows))
        # ... processing logic

    # In the API
    @app.get("/alerts/{id}")
    async def get_alert(id: str):
        with start_span("get_alert") as span:
            span.set_attribute("alert_id", id)
            # ... logic
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# Configuration
OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "amastan-fraud-shield")
OTEL_ENVIRONMENT = os.getenv("OTEL_ENVIRONMENT", "development")


class TracingNotAvailable(Exception):
    pass


class Span:
    """Simple span representation when OpenTelemetry is not configured."""

    def __init__(self, name: str):
        self.name = name
        self.attributes = {}
        self.events = []

    def set_attribute(self, key: str, value):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict] = None):
        self.events.append({"name": name, "attributes": attributes or {}})

    def record_exception(self, exception: Exception):
        self.add_event("exception", {"type": type(exception).__name__, "message": str(exception)})


class AmastanTracer:
    """
    Distributed tracing wrapper for the fraud detection pipeline.
    Gracefully falls back to no-op spans if OpenTelemetry is not configured.
    """

    def __init__(self):
        self._provider = None
        self._tracer = None
        self._is_configured = False

    def initialize(self):
        """Initialize OpenTelemetry SDK."""
        if self._is_configured:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Create provider
            provider = TracerProvider()
            self._provider = provider

            # Configure OTLP exporter
            exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_ENDPOINT, insecure=True)
            span_processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(span_processor)

            # Set as global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer
            self._tracer = trace.get_tracer(OTEL_SERVICE_NAME)
            self._is_configured = True

            logger.info(f"OpenTelemetry tracer initialized: {OTEL_EXPORTER_ENDPOINT}")

        except ImportError:
            logger.warning("OpenTelemetry packages not installed. Using no-op tracer.")
        except Exception as e:
            logger.warning(f"OpenTelemetry initialization failed: {e}. Using no-op tracer.")

    @contextmanager
    def start_span(self, name: str, parent=None) -> Generator[Span, None, None]:
        """
        Start a new span. Works with both OTel and no-op fallback.

        Usage:
            with tracer.start_span("process_batch") as span:
                span.set_attribute("batch_size", 100)
        """
        if self._is_configured and self._tracer:
            from opentelemetry import trace

            ctx = trace.set_span_in_context(parent) if parent else None
            with self._tracer.start_as_current_span(name, context=ctx) as otel_span:
                span = Span(name)
                span._otel_span = otel_span

                # Patch set_attribute to also set on OTel span
                original_set_attr = span.set_attribute

                def patched_set_attr(key, value):
                    original_set_attr(key, value)
                    otel_span.set_attribute(key, value)

                span.set_attribute = patched_set_attr

                yield span
        else:
            # No-op fallback
            span = Span(name)
            yield span

    @contextmanager
    def start_pipeline_span(self, tx_id: str) -> Generator[Span, None, None]:
        """
        Start a pipeline-level span that tracks the entire journey of a transaction.
        From Kafka ingest -> quality gates -> ML scoring -> alert dispatch.
        """
        with self.start_span(f"fraud_pipeline.{tx_id}") as span:
            span.set_attribute("transaction_id", tx_id)
            span.set_attribute("service", OTEL_SERVICE_NAME)
            span.set_attribute("environment", OTEL_ENVIRONMENT)
            yield span

    def record_metric(self, name: str, value: float, attributes: Optional[dict] = None):
        """Record a metric value via OpenTelemetry."""
        if self._is_configured:
            try:
                from opentelemetry.metrics import set_meter_provider
                from opentelemetry.sdk.metrics import MeterProvider

                # Metric recording would go here
                pass
            except Exception:
                pass


# Singleton tracer
tracer = AmastanTracer()


def initialize_tracing():
    """Initialize tracing at application startup."""
    tracer.initialize()


@contextmanager
def start_span(name: str) -> Generator[Span, None, None]:
    """Module-level convenience function for starting spans."""
    with tracer.start_span(name) as span:
        yield span


def trace_api_request(func):
    """Decorator for tracing API requests."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        with tracer.start_span(f"api.{func.__name__}") as span:
            if "request" in kwargs:
                span.set_attribute("method", kwargs["request"].method)
                span.set_attribute("path", kwargs["request"].url.path)
            try:
                result = await func(*args, **kwargs)
                span.set_attribute("status", "success")
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("status", "error")
                raise

    return wrapper


def trace_batch_processing(func):
    """Decorator for tracing Spark batch processing."""
    import functools

    @functools.wraps(func)
    def wrapper(batch_df, epoch_id):
        batch_count = batch_df.count() if batch_df else 0
        with tracer.start_span(f"spark.batch.{epoch_id}") as span:
            span.set_attribute("epoch_id", epoch_id)
            span.set_attribute("batch_size", batch_count)
            try:
                result = func(batch_df, epoch_id)
                span.set_attribute("status", "success")
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("status", "error")
                raise

    return wrapper
