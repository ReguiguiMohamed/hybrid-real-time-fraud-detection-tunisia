"""Isolation Forest anomaly detector for zero-day fraud signals."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.utils import get_sqlite_connection


DEFAULT_ANOMALY_FEATURES = [
    "amount_tnd",
    "ml_probability",
    "risk_score",
    "v_count",
    "g_dist",
]


class IsolationForestAnomalyDetector:
    """Train and score an unsupervised anomaly model beside the supervised fraud model."""

    HIGH_ANOMALY_THRESHOLD = -0.3

    def __init__(
        self,
        feature_columns: Optional[list[str]] = None,
        contamination: float = 0.02,
        random_state: int = 42,
    ):
        self.feature_columns = feature_columns or list(DEFAULT_ANOMALY_FEATURES)
        self.contamination = contamination
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.model_version: Optional[str] = None

    def _coerce_features(self, records: pd.DataFrame | Iterable[dict]) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            df = records.copy()
        else:
            df = pd.DataFrame(list(records))

        for column in self.feature_columns:
            if column not in df.columns:
                df[column] = np.nan
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df[self.feature_columns]

    def train(self, records: pd.DataFrame | Iterable[dict], model_version: Optional[str] = None):
        features = self._coerce_features(records)
        if len(features) < 2:
            raise ValueError("Isolation Forest training requires at least 2 records")

        self.pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=self.random_state,
            )),
        ])
        self.pipeline.fit(features)
        self.model_version = model_version or f"isoforest_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        return self

    def score(self, records: pd.DataFrame | Iterable[dict]) -> list[dict]:
        if self.pipeline is None:
            raise RuntimeError("Isolation Forest model is not trained or loaded")

        features = self._coerce_features(records)
        scores = self.pipeline.decision_function(features)
        predictions = self.pipeline.predict(features)
        results = []
        for score, prediction in zip(scores, predictions):
            score_value = round(float(score), 6)
            results.append({
                "anomaly_score": score_value,
                "is_anomaly": bool(prediction == -1),
                "alert_type": "HIGH_ANOMALY" if score_value < self.HIGH_ANOMALY_THRESHOLD else None,
                "model_version": self.model_version,
            })
        return results

    def save(self, model_path: str | Path) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained Isolation Forest model")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "feature_columns": self.feature_columns,
            "contamination": self.contamination,
            "model_version": self.model_version,
        }, path)
        return path

    @classmethod
    def load(cls, model_path: str | Path) -> "IsolationForestAnomalyDetector":
        payload = joblib.load(model_path)
        detector = cls(
            feature_columns=payload["feature_columns"],
            contamination=payload["contamination"],
        )
        detector.pipeline = payload["pipeline"]
        detector.model_version = payload.get("model_version")
        return detector

    @classmethod
    def load_training_data_from_db(cls, db_path: str, limit: int = 10000) -> pd.DataFrame:
        conn = get_sqlite_connection(db_path)
        try:
            query = """
                SELECT amount_tnd, ml_probability, risk_score, v_count, g_dist
                FROM high_risk_alerts
                ORDER BY created_at DESC
                LIMIT ?
            """
            try:
                return pd.read_sql_query(query, conn, params=(limit,))
            except Exception:
                fallback_query = """
                    SELECT amount_tnd, ml_probability,
                           ml_probability AS risk_score,
                           NULL AS v_count,
                           NULL AS g_dist
                    FROM high_risk_alerts
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                return pd.read_sql_query(fallback_query, conn, params=(limit,))
        finally:
            conn.close()

    def metadata_json(self) -> str:
        return json.dumps({
            "model_version": self.model_version,
            "feature_columns": self.feature_columns,
            "contamination": self.contamination,
            "high_anomaly_threshold": self.HIGH_ANOMALY_THRESHOLD,
        }, sort_keys=True)
