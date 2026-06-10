"""
Shadow Model Deployment for Amastan Fraud Shield Guard

Runs a challenger model in "shadow mode" alongside the production champion.
Shadow model scores every transaction but DOES NOT trigger alerts.
Its predictions are logged and compared against the champion for offline evaluation.

This prevents unproven models from blocking real transactions while proving
their superiority on live data before promotion.

Usage:
    from src.ml.shadow_model import ShadowModelManager

    shadow = ShadowModelManager()
    shadow.register_shadow_model("models/registry/fraud_xgb_shadow_v2")

    champion_score, shadow_score = shadow.score_with_both(df)

    results = shadow.compare_performance()
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import and_

from shared.database import SessionLocal, ShadowModelRegistry, ShadowScoreLog

logger = logging.getLogger(__name__)


class ShadowModelManager:
    """
    Manages shadow model deployment alongside the champion model.
    Persistence uses the shared SQLAlchemy database (SQLite or PostgreSQL).
    """

    def __init__(self, db_session=None):
        self._shadow_model = None
        self._shadow_model_path = None
        self._shadow_registered_at = None
        self._db = db_session

    def _db_session(self):
        if self._db is not None:
            return self._db
        return SessionLocal()

    def _close_session(self, session):
        if self._db is None and session is not None:
            session.close()

    def _get_active_shadow(self, session):
        return session.query(ShadowModelRegistry).filter(ShadowModelRegistry.status == "shadow").first()

    def register_shadow_model(self, model_path: str, version_id: str = None) -> bool:
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Shadow model path not found: {model_path}")
            return False

        if version_id is None:
            version_id = f"shadow_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        try:
            from xgboost.spark import SparkXGBClassifierModel

            self._shadow_model = SparkXGBClassifierModel.load(str(path))
            logger.info(f"Shadow model loaded (Spark XGBoost): {path}")
        except Exception:
            try:
                import joblib

                self._shadow_model = joblib.load(str(path / "pipeline.pkl"))
                logger.info(f"Shadow model loaded (sklearn pipeline): {path}")
            except Exception as e:
                logger.error(f"Could not load shadow model: {e}")
                return False

        self._shadow_model_path = str(path)

        session = self._db_session()
        try:
            session.query(ShadowModelRegistry).filter(ShadowModelRegistry.status == "shadow").update(
                {
                    "unregistered_at": datetime.now(timezone.utc),
                    "status": "inactive",
                }
            )

            entry = ShadowModelRegistry(
                version_id=version_id,
                model_path=str(path),
                status="shadow",
            )
            session.add(entry)
            session.commit()

            self._shadow_registered_at = datetime.now(timezone.utc)
            logger.info(f"Shadow model registered: {version_id} from {path}")
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session)

    def unregister_shadow_model(self):
        self._shadow_model = None
        self._shadow_model_path = None

        session = self._db_session()
        try:
            session.query(ShadowModelRegistry).filter(ShadowModelRegistry.status == "shadow").update(
                {
                    "unregistered_at": datetime.now(timezone.utc),
                    "status": "inactive",
                }
            )
            session.commit()
            logger.info("Shadow model unregistered")
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session)

    def score_with_both(self, spark_df, features_col: str = "features") -> tuple:
        if self._shadow_model is None:
            return spark_df, None

        try:
            from pyspark.ml.feature import VectorAssembler
            from pyspark.sql.functions import col

            if hasattr(self._shadow_model, "transform"):
                shadow_scored = self._shadow_model.transform(spark_df)
                if "probability" in shadow_scored.columns:
                    shadow_scored = shadow_scored.withColumnRenamed("probability", "shadow_probability")
                if "prediction" in shadow_scored.columns:
                    shadow_scored = shadow_scored.withColumnRenamed("prediction", "shadow_prediction")
                return spark_df, shadow_scored
            else:
                logger.warning("Shadow model is sklearn type, cannot score Spark DataFrame directly")
                return spark_df, None
        except Exception as e:
            logger.error(f"Shadow model scoring failed: {e}")
            return spark_df, None

    def record_shadow_comparison(
        self,
        tx_id: str,
        champion_prob: float,
        shadow_prob: float,
        champion_label: int = None,
        shadow_label: int = None,
        analyst_label: str = None,
        latency_ms: float = None,
    ):
        try:
            session = self._db_session()
            entry = ShadowScoreLog(
                transaction_id=tx_id,
                champion_score=champion_prob,
                shadow_score=shadow_prob,
                score_diff=abs(champion_prob - shadow_prob),
                champion_label=champion_label if champion_label is not None else (1 if champion_prob > 0.5 else 0),
                shadow_label=shadow_label if shadow_label is not None else (1 if shadow_prob > 0.5 else 0),
                analyst_label=analyst_label,
                latency_ms=latency_ms,
            )
            session.add(entry)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session)

    def compare_performance(self, window_hours: int = 24) -> dict:
        session = self._db_session()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
            rows = session.query(ShadowScoreLog).filter(ShadowScoreLog.timestamp > cutoff).all()

            if not rows:
                return {
                    "status": "insufficient_data",
                    "comparisons": 0,
                    "message": f"No shadow comparisons in the last {window_hours} hours",
                }

            champion_scores = [r.champion_score for r in rows]
            shadow_scores = [r.shadow_score for r in rows]
            score_diffs = [r.score_diff for r in rows]

            champion_mean = float(np.mean(champion_scores))
            shadow_mean = float(np.mean(shadow_scores))
            mean_diff = float(abs(champion_mean - shadow_mean))

            champion_labels = [
                r.champion_label if r.champion_label is not None else (1 if r.champion_score > 0.5 else 0) for r in rows
            ]
            shadow_labels = [
                r.shadow_label if r.shadow_label is not None else (1 if r.shadow_score > 0.5 else 0) for r in rows
            ]
            disagreements = sum(1 for c, s in zip(champion_labels, shadow_labels) if c != s)
            disagreement_rate = disagreements / max(len(rows), 1)

            high_divergence = sum(1 for d in score_diffs if d > 0.2)
            high_divergence_rate = high_divergence / max(len(rows), 1)

            labelled = [r for r in rows if r.analyst_label is not None]
            label_agreement_rate = None
            shadow_correct_count = None
            champion_correct_count = None
            if labelled:
                total_labelled = len(labelled)
                champion_correct = sum(
                    1
                    for r in labelled
                    if (r.champion_label is not None and str(r.champion_label) == r.analyst_label)
                    or (r.champion_label is None and (1 if r.champion_score > 0.5 else 0) == int(r.analyst_label))
                )
                shadow_correct = sum(
                    1
                    for r in labelled
                    if (r.shadow_label is not None and str(r.shadow_label) == r.analyst_label)
                    or (r.shadow_label is None and (1 if r.shadow_score > 0.5 else 0) == int(r.analyst_label))
                )
                label_agreement_rate = champion_correct / total_labelled
                shadow_correct_count = shadow_correct
                champion_correct_count = champion_correct

            latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
            avg_latency_ms = float(np.mean(latencies)) if latencies else None
            p95_latency_ms = float(np.percentile(latencies, 95)) if latencies else None

            active_shadow = self._get_active_shadow(session)
            if active_shadow:
                active_shadow.total_comparisons = len(rows)
                active_shadow.avg_score_diff = mean_diff
                shadow_wins = sum(1 for r in rows if r.shadow_score > r.champion_score)
                active_shadow.shadow_wins = shadow_wins
                active_shadow.champion_wins = len(rows) - shadow_wins
                active_shadow.total_samples = len(rows)
                if avg_latency_ms is not None:
                    active_shadow.avg_latency_ms = avg_latency_ms
                session.commit()

            results = {
                "status": "comparing",
                "comparisons": len(rows),
                "window_hours": window_hours,
                "champion_mean_score": champion_mean,
                "shadow_mean_score": shadow_mean,
                "mean_score_difference": mean_diff,
                "disagreement_rate": disagreement_rate,
                "high_divergence_rate": high_divergence_rate,
                "recommendation": self._generate_recommendation(disagreement_rate, high_divergence_rate, mean_diff),
                "label_analysis": (
                    {
                        "labelled_count": len(labelled),
                        "label_agreement_rate": label_agreement_rate,
                        "champion_correct_count": champion_correct_count,
                        "shadow_correct_count": shadow_correct_count,
                    }
                    if labelled
                    else None
                ),
                "latency": (
                    {
                        "avg_ms": avg_latency_ms,
                        "p95_ms": p95_latency_ms,
                    }
                    if avg_latency_ms is not None
                    else None
                ),
            }

            logger.info(
                f"Shadow comparison ({len(rows)} samples, {len(labelled)} labelled): " f"{results['recommendation']}"
            )
            return results
        finally:
            self._close_session(session)

    @staticmethod
    def _generate_recommendation(disagreement_rate: float, high_divergence_rate: float, mean_diff: float) -> str:
        if disagreement_rate > 0.3:
            return "HOLD - Shadow model disagrees with champion too often. Requires analyst review."
        elif high_divergence_rate > 0.1:
            return "REVIEW - Significant score divergence on subset. Check feature engineering differences."
        elif mean_diff < 0.02 and disagreement_rate < 0.05:
            return "ALIGNED - Similar behavior observed; labeled evaluation is still required."
        else:
            return "MONITOR - Continue collecting comparison data before promotion decision."

    def has_shadow_model(self) -> bool:
        return self._shadow_model is not None

    def get_shadow_status(self) -> dict:
        session = self._db_session()
        try:
            row = session.query(ShadowModelRegistry).filter(ShadowModelRegistry.status == "shadow").first()

            if not row:
                return {"active": False, "message": "No shadow model registered"}

            return {
                "active": True,
                "version_id": row.version_id,
                "model_path": row.model_path,
                "registered_at": row.registered_at,
                "total_comparisons": row.total_comparisons,
                "avg_score_diff": row.avg_score_diff,
                "total_samples": row.total_samples or 0,
                "avg_latency_ms": row.avg_latency_ms,
            }
        finally:
            self._close_session(session)

    def record_batch_comparison(
        self,
        comparisons: list,
    ):
        """Record multiple shadow comparisons in a single transaction.

        Args:
            comparisons: List of dicts with keys:
                transaction_id, champion_score, shadow_score,
                champion_label, shadow_label, analyst_label, latency_ms
        """
        session = self._db_session()
        try:
            entries = [
                ShadowScoreLog(
                    transaction_id=c["transaction_id"],
                    champion_score=c["champion_score"],
                    shadow_score=c["shadow_score"],
                    score_diff=abs(c["champion_score"] - c["shadow_score"]),
                    champion_label=c.get("champion_label"),
                    shadow_label=c.get("shadow_label"),
                    analyst_label=c.get("analyst_label"),
                    latency_ms=c.get("latency_ms"),
                )
                for c in comparisons
            ]
            session.bulk_save_objects(entries)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._close_session(session)
