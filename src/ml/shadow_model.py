"""
Shadow Model Deployment for Amastan Fraud Shield Guard

Runs a challenger model in "shadow mode" alongside the production champion.
Shadow model scores every transaction but DOES NOT trigger alerts.
Its predictions are logged and compared against the champion for offline evaluation.

This prevents unproven models from blocking real transactions while proving
their superiority on live data before promotion.

Usage:
    # In the consumer, shadow scoring is automatic if a shadow model is registered
    from src.ml.shadow_model import ShadowModelManager

    shadow = ShadowModelManager()
    shadow.register_shadow_model("models/registry/fraud_xgb_shadow_v2")

    # During scoring:
    champion_score, shadow_score = shadow.score_with_both(df)

    # Evaluate shadow vs champion on accumulated data
    results = shadow.compare_performance()
"""
import json
import logging
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ShadowModelManager:
    """
    Manages shadow model deployment alongside the champion model.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │              Streaming Batch                     │
    │                                                  │
    │  Transaction ──► Champion Model ──► Score ──► Alert if high
    │                      │                          │
    │                      └──► Shadow Model ──► Score (logged only)
    │                                                  │
    │  Comparison Engine: Daily evaluation of shadow   │
    │  vs champion on precision, recall, F1, latency   │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, feedback_db_path: str = "./data/feedback.db"):
        self.feedback_db_path = feedback_db_path
        self._shadow_model = None
        self._shadow_model_path = None
        self._shadow_registered_at = None
        self._shadow_scores = []  # Accumulate (tx_id, champion_score, shadow_score)

    def _ensure_shadow_model_table(self):
        """Ensure the shadow model registry table exists."""
        conn = sqlite3.connect(self.feedback_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_model_registry (
                version_id TEXT PRIMARY KEY,
                model_path TEXT NOT NULL,
                registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                unregistered_at DATETIME,
                status TEXT DEFAULT 'shadow',
                total_comparisons INTEGER DEFAULT 0,
                avg_score_diff REAL DEFAULT 0.0,
                shadow_wins INTEGER DEFAULT 0,
                champion_wins INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_score_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                champion_score REAL NOT NULL,
                shadow_score REAL NOT NULL,
                score_diff REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_tx ON shadow_score_log(transaction_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_ts ON shadow_score_log(timestamp)")
        conn.commit()
        conn.close()

    def register_shadow_model(self, model_path: str, version_id: str = None) -> bool:
        """
        Register a shadow model for parallel scoring.

        Args:
            model_path: Path to the model directory or file
            version_id: Unique identifier for the shadow model

        Returns:
            True if registration succeeded.
        """
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Shadow model path not found: {model_path}")
            return False

        if version_id is None:
            version_id = f"shadow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Try loading as Spark XGBoost model
            from xgboost.spark import SparkXGBClassifierModel
            self._shadow_model = SparkXGBClassifierModel.load(str(path))
            logger.info(f"Shadow model loaded (Spark XGBoost): {path}")
        except Exception:
            try:
                # Try loading as sklearn pipeline
                import joblib
                self._shadow_model = joblib.load(str(path / "pipeline.pkl"))
                logger.info(f"Shadow model loaded (sklearn pipeline): {path}")
            except Exception as e:
                logger.error(f"Could not load shadow model: {e}")
                return False

        self._shadow_model_path = str(path)

        # Register in SQLite
        self._ensure_shadow_model_table()
        conn = sqlite3.connect(self.feedback_db_path)
        cursor = conn.cursor()

        # Unregister any active shadow models
        cursor.execute(
            "UPDATE shadow_model_registry SET unregistered_at = CURRENT_TIMESTAMP, status = 'inactive' WHERE status = 'shadow'"
        )

        # Register new shadow model
        cursor.execute("""
            INSERT INTO shadow_model_registry (version_id, model_path, status)
            VALUES (?, ?, 'shadow')
        """, (version_id, str(path)))
        conn.commit()
        conn.close()

        self._shadow_registered_at = datetime.utcnow()
        logger.info(f"Shadow model registered: {version_id} from {path}")
        return True

    def unregister_shadow_model(self):
        """Remove the current shadow model."""
        self._shadow_model = None
        self._shadow_model_path = None

        conn = sqlite3.connect(self.feedback_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE shadow_model_registry SET unregistered_at = CURRENT_TIMESTAMP, status = 'inactive' WHERE status = 'shadow'"
        )
        conn.commit()
        conn.close()
        logger.info("Shadow model unregistered")

    def score_with_both(self, spark_df, features_col: str = "features") -> tuple:
        """
        Score a Spark DataFrame with both champion and shadow models.

        Returns:
            (champion_scored_df, shadow_scored_df)
        """
        if self._shadow_model is None:
            return spark_df, None

        try:
            from pyspark.ml.feature import VectorAssembler
            from pyspark.sql.functions import col

            # Shadow model scoring
            if hasattr(self._shadow_model, "transform"):
                # Spark model
                shadow_scored = self._shadow_model.transform(spark_df)
                if "probability" in shadow_scored.columns:
                    shadow_scored = shadow_scored.withColumnRenamed("probability", "shadow_probability")
                if "prediction" in shadow_scored.columns:
                    shadow_scored = shadow_scored.withColumnRenamed("prediction", "shadow_prediction")
                return spark_df, shadow_scored
            else:
                # sklearn model (needs pandas conversion)
                logger.warning("Shadow model is sklearn type, cannot score Spark DataFrame directly")
                return spark_df, None

        except Exception as e:
            logger.error(f"Shadow model scoring failed: {e}")
            return spark_df, None

    def record_shadow_comparison(self, tx_id: str, champion_prob: float, shadow_prob: float):
        """Record a single shadow comparison for later analysis."""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO shadow_score_log (transaction_id, champion_score, shadow_score, score_diff)
                VALUES (?, ?, ?, ?)
            """, (tx_id, champion_prob, shadow_prob, abs(champion_prob - shadow_prob)))
            conn.commit()
            conn.close()
            self._shadow_scores.append((tx_id, champion_prob, shadow_prob))
        except Exception as e:
            logger.error(f"Failed to record shadow comparison: {e}")

    def compare_performance(self, window_hours: int = 24) -> dict:
        """
        Compare shadow model vs champion on accumulated data.

        Returns metrics that determine whether the shadow should be promoted.
        """
        self._ensure_shadow_model_table()
        conn = sqlite3.connect(self.feedback_db_path)
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()

        # Get all comparisons in the window
        cursor.execute("""
            SELECT champion_score, shadow_score, score_diff
            FROM shadow_score_log
            WHERE timestamp > ?
        """, (cutoff,))
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return {
                "status": "insufficient_data",
                "comparisons": 0,
                "message": f"No shadow comparisons in the last {window_hours} hours",
            }

        champion_scores = [r[0] for r in rows]
        shadow_scores = [r[1] for r in rows]
        score_diffs = [r[2] for r in rows]

        # Statistical comparison
        champion_mean = np.mean(champion_scores)
        shadow_mean = np.mean(shadow_scores)
        mean_diff = abs(champion_mean - shadow_mean)

        # How often do they disagree on classification (>0.5 threshold)?
        champion_labels = [1 if s > 0.5 else 0 for s in champion_scores]
        shadow_labels = [1 if s > 0.5 else 0 for s in shadow_scores]
        disagreements = sum(1 for c, s in zip(champion_labels, shadow_labels) if c != s)
        disagreement_rate = disagreements / max(len(rows), 1)

        # Score divergence analysis
        high_divergence = sum(1 for d in score_diffs if d > 0.2)
        high_divergence_rate = high_divergence / max(len(rows), 1)

        # Update shadow model registry stats
        cursor.execute("""
            SELECT version_id FROM shadow_model_registry WHERE status = 'shadow' LIMIT 1
        """)
        shadow_row = cursor.fetchone()
        if shadow_row:
            cursor.execute("""
                UPDATE shadow_model_registry
                SET total_comparisons = ?,
                    avg_score_diff = ?
                WHERE version_id = ?
            """, (len(rows), float(mean_diff), shadow_row[0]))

        conn.commit()
        conn.close()

        results = {
            "status": "comparing",
            "comparisons": len(rows),
            "window_hours": window_hours,
            "champion_mean_score": float(champion_mean),
            "shadow_mean_score": float(shadow_mean),
            "mean_score_difference": float(mean_diff),
            "disagreement_rate": float(disagreement_rate),
            "high_divergence_rate": float(high_divergence_rate),
            "recommendation": self._generate_recommendation(
                disagreement_rate, high_divergence_rate, mean_diff
            ),
        }

        logger.info(f"Shadow comparison ({len(rows)} samples): {results['recommendation']}")
        return results

    @staticmethod
    def _generate_recommendation(disagreement_rate: float, high_divergence_rate: float, mean_diff: float) -> str:
        """Generate a recommendation based on comparison metrics."""
        if disagreement_rate > 0.3:
            return "HOLD - Shadow model disagrees with champion too often. Requires analyst review."
        elif high_divergence_rate > 0.1:
            return "REVIEW - Significant score divergence on subset. Check feature engineering differences."
        elif mean_diff < 0.02 and disagreement_rate < 0.05:
            return "SAFE_TO_PROMOTE - Shadow model closely matches champion behavior."
        else:
            return "MONITOR - Continue collecting comparison data before promotion decision."

    def has_shadow_model(self) -> bool:
        """Check if a shadow model is currently active."""
        return self._shadow_model is not None

    def get_shadow_status(self) -> dict:
        """Get current shadow model status."""
        self._ensure_shadow_model_table()
        conn = sqlite3.connect(self.feedback_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT version_id, model_path, registered_at, total_comparisons, avg_score_diff FROM shadow_model_registry WHERE status = 'shadow'"
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {"active": False, "message": "No shadow model registered"}

        return {
            "active": True,
            "version_id": row[0],
            "model_path": row[1],
            "registered_at": row[2],
            "total_comparisons": row[3],
            "avg_score_diff": row[4],
        }
