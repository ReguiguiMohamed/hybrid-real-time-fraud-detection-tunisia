"""
Structured JSON logging configuration for the fraud detection system.
Provides consistent log format across all services (producer, consumer, API, dashboard).
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include any extra fields passed via `extra={}` in logging calls
        for key in ("transaction_id", "user_id", "alert_type", "latency_ms", "service"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(service_name: str = "fraud-detection", level: str = "INFO") -> None:
    """
    Configure root logger with structured JSON output.

    Args:
        service_name: Name of the service (included in every log line).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers if called more than once
    if any(
        isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JSONFormatter) for h in root_logger.handlers
    ):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(handler)

    # Inject service name into all records via a filter
    class ServiceFilter(logging.Filter):
        def filter(self, record):
            record.service = service_name
            return True

    root_logger.addFilter(ServiceFilter())

    # Reduce noise from third-party libraries
    for noisy in ("urllib3", "kafka", "chromadb", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
