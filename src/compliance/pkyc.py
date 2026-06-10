"""Perpetual KYC trigger evaluation and publishing."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from shared.pii_masking import hash_pii
from shared.utils import get_sqlite_connection

logger = logging.getLogger(__name__)


class PKYCTriggerEvent(BaseModel):
    event_type: str = "pKYC_trigger"
    account_id: str = Field(..., description="HMAC-SHA256 masked account/user identifier")
    trigger_reason: str
    timestamp: str
    current_risk_tier: str
    signals: dict[str, Any]
    transaction_id: Optional[str] = None


class PKYCTriggerEvaluator:
    """Evaluate transaction/alert facts against pKYC trigger rules."""

    SCORE_SPIKE_THRESHOLD = 0.7
    LOW_RISK_SCORE_MAX = 0.3
    VELOCITY_WOW_TRIGGER_PCT = 300.0

    @staticmethod
    def risk_tier(score: float) -> str:
        if score >= 0.85:
            return "CRITICAL"
        if score >= 0.7:
            return "HIGH"
        if score >= 0.3:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _as_float(value, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @classmethod
    def _velocity_wow_pct(cls, tx_data: dict) -> Optional[float]:
        explicit = cls._as_float(tx_data.get("velocity_week_over_week_pct"))
        if explicit is not None:
            return explicit

        current = cls._as_float(tx_data.get("current_week_tx_count"))
        previous = cls._as_float(tx_data.get("previous_week_tx_count"))
        if current is None or previous is None or previous <= 0:
            return None
        return ((current - previous) / previous) * 100.0

    @classmethod
    def evaluate(cls, tx_data: dict, ml_probability: float) -> Optional[PKYCTriggerEvent]:
        account_source = (
            tx_data.get("account_id")
            or tx_data.get("user_id")
            or tx_data.get("sender_account")
            or tx_data.get("source_account")
        )
        if not account_source:
            return None

        reasons = []
        signals: dict[str, Any] = {
            "ml_probability": round(float(ml_probability), 6),
        }

        previous_tier = str(tx_data.get("previous_risk_tier", "")).upper()
        previous_score = cls._as_float(tx_data.get("previous_ml_probability"))
        was_low_risk = previous_tier == "LOW" or (
            previous_score is not None and previous_score <= cls.LOW_RISK_SCORE_MAX
        )
        if ml_probability > cls.SCORE_SPIKE_THRESHOLD and was_low_risk:
            reasons.append("LOW_RISK_TO_HIGH_SCORE")
            signals["previous_risk_tier"] = previous_tier or cls.risk_tier(previous_score or 0.0)
            if previous_score is not None:
                signals["previous_ml_probability"] = round(previous_score, 6)

        account_event_type = str(tx_data.get("account_event_type", "")).upper()
        if str(tx_data.get("account_type", "")).upper() == "FCY" and (
            account_event_type == "FCY_ACCOUNT_OPENED"
            or cls._as_bool(tx_data.get("fcy_account_opened"))
            or cls._as_float(tx_data.get("account_age_days")) == 0
        ):
            reasons.append("FCY_ACCOUNT_OPENED")
            signals["account_type"] = "FCY"
            if tx_data.get("fcy_currency"):
                signals["fcy_currency"] = tx_data.get("fcy_currency")

        velocity_wow_pct = cls._velocity_wow_pct(tx_data)
        if velocity_wow_pct is not None and velocity_wow_pct > cls.VELOCITY_WOW_TRIGGER_PCT:
            reasons.append("VELOCITY_WOW_SPIKE")
            signals["velocity_week_over_week_pct"] = round(velocity_wow_pct, 2)

        if cls._as_bool(tx_data.get("transaction_ring_detected")) or cls._as_bool(tx_data.get("gnn_ring_detected")):
            reasons.append("TRANSACTION_RING_DETECTED")
            if tx_data.get("gnn_cluster_id"):
                signals["gnn_cluster_id"] = tx_data.get("gnn_cluster_id")

        if not reasons:
            return None

        return PKYCTriggerEvent(
            account_id=hash_pii(str(account_source)),
            trigger_reason="|".join(reasons),
            timestamp=datetime.now(timezone.utc).isoformat(),
            current_risk_tier=cls.risk_tier(float(ml_probability)),
            signals=signals,
            transaction_id=tx_data.get("transaction_id"),
        )


class PKYCPublisher:
    """Publish pKYC trigger events to Kafka without leaking raw account IDs."""

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        producer=None,
        audit_db_path: Optional[str] = None,
    ):
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
        self.topic = topic or os.getenv("PKYC_TOPIC", "pkyc_triggers")
        self.audit_db_path = audit_db_path or os.getenv("FEEDBACK_DB_PATH", "./data/feedback.db")
        self._producer = producer
        self._producer_init_attempted = producer is not None

    def _record_event(self, event: PKYCTriggerEvent) -> None:
        conn = None
        try:
            conn = get_sqlite_connection(self.audit_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pkyc_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    current_risk_tier TEXT NOT NULL,
                    signals TEXT NOT NULL,
                    transaction_id TEXT
                )
            """)
            cursor.execute(
                """
                INSERT INTO pkyc_triggers
                (event_type, account_id, trigger_reason, timestamp, current_risk_tier, signals, transaction_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_type,
                    event.account_id,
                    event.trigger_reason,
                    event.timestamp,
                    event.current_risk_tier,
                    json.dumps(event.signals, sort_keys=True, default=str),
                    event.transaction_id,
                ),
            )
            conn.commit()
        except Exception:
            logger.exception("Failed to record pKYC trigger audit event for tx %s", event.transaction_id)
        finally:
            if conn is not None:
                conn.close()

    def _get_producer(self):
        if self._producer_init_attempted:
            return self._producer

        self._producer_init_attempted = True
        try:
            from confluent_kafka import Producer

            self._producer = Producer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "client.id": "fraud-pkyc-publisher",
                }
            )
        except Exception as exc:
            logger.warning("pKYC publisher unavailable: %s", exc)
            self._producer = None
        return self._producer

    def publish_for_transaction(self, tx_data: dict, ml_probability: float) -> Optional[PKYCTriggerEvent]:
        event = PKYCTriggerEvaluator.evaluate(tx_data, ml_probability)
        if event is None:
            return None
        self._record_event(event)

        producer = self._get_producer()
        if producer is None:
            logger.warning(
                "pKYC trigger evaluated but not published; Kafka producer unavailable for tx %s",
                event.transaction_id,
            )
            return event

        payload = event.model_dump_json().encode("utf-8")
        key = event.account_id.encode("utf-8")
        producer.produce(self.topic, key=key, value=payload)
        producer.poll(0)
        logger.info("Published pKYC trigger %s for tx %s", event.trigger_reason, event.transaction_id)
        return event
