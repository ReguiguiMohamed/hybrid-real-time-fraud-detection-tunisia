# dashboard/monitoring.py
import os
import statistics
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np

from compliance.change_audit import append_change_audit_event
from shared.utils import get_sqlite_connection


class ForensicAnalyticEngine:
    PSI_RETRAIN_THRESHOLD = 0.2
    PAGE_HINKLEY_DELTA = 0.005
    PAGE_HINKLEY_LAMBDA = 0.05

    def __init__(self, db_path: Path = Path("./data/feedback.db")):
        self.db_path = db_path
        self.inference_latencies = deque(maxlen=1000)  # Keep last 1000 measurements
        self.drift_monitoring = {}

    def record_inference_latency(self, latency_ms: float):
        """Record the latency of an inference call"""
        self.inference_latencies.append(latency_ms)

    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.inference_latencies:
            return {"avg_latency_ms": 0, "p95_latency_ms": 0, "p99_latency_ms": 0, "total_calls": 0}

        latencies = list(self.inference_latencies)
        return {
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "p99_latency_ms": round(np.percentile(latencies, 99), 2),
            "total_calls": len(latencies),
        }

    def get_feedback_analysis(self):
        """Analyze feedback patterns"""
        try:
            conn = get_sqlite_connection(str(self.db_path))
            cursor = conn.cursor()

            # Get feedback counts
            cursor.execute("""
                SELECT analyst_label, COUNT(*) as count
                FROM feedback_labels
                GROUP BY analyst_label
            """)

            feedback_counts = dict(cursor.fetchall())

            # Calculate precision based on feedback
            confirmed_fraud = feedback_counts.get("Confirmed Fraud", 0)
            total_labeled = sum(feedback_counts.values())

            precision = confirmed_fraud / total_labeled if total_labeled > 0 else 0

            # Get ML probability distribution for confirmed fraud vs false positives
            cursor.execute("""
                SELECT hra.ml_probability, fl.analyst_label
                FROM high_risk_alerts hra
                JOIN feedback_labels fl ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
            """)

            prob_label_pairs = cursor.fetchall()

            conn.close()

            return {
                "precision": round(precision, 3),
                "feedback_counts": feedback_counts,
                "total_feedback": total_labeled,
                "prob_label_pairs": prob_label_pairs,
            }
        except Exception as e:
            print(f"Error getting feedback analysis: {e}")
            return {"precision": 0, "feedback_counts": {}, "total_feedback": 0, "prob_label_pairs": []}

    def get_ml_threshold_analysis(self):
        """Analyze optimal threshold based on feedback"""
        analysis = self.get_feedback_analysis()
        prob_label_pairs = analysis.get("prob_label_pairs", [])

        if not prob_label_pairs:
            return {"optimal_threshold": 0.85, "threshold_analysis": {}}

        # Calculate precision at different thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        threshold_analysis = {}

        for threshold in thresholds:
            tp = sum(1 for prob, label in prob_label_pairs if prob >= threshold and label == "Confirmed Fraud")
            fp = sum(1 for prob, label in prob_label_pairs if prob >= threshold and label == "False Positive")

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = (
                tp / sum(1 for _, label in prob_label_pairs if label == "Confirmed Fraud")
                if sum(1 for _, label in prob_label_pairs if label == "Confirmed Fraud") > 0
                else 0
            )

            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            threshold_analysis[threshold] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1_score, 3),
            }

        # Find threshold with highest F1 score
        best_threshold = max(thresholds, key=lambda t: threshold_analysis[t]["f1_score"])

        return {"optimal_threshold": best_threshold, "threshold_analysis": threshold_analysis}

    def detect_feature_drift(
        self, feature_name: str, current_values: List[float], reference_values: List[float] = None
    ):
        """Detect statistical drift in feature distributions"""
        if reference_values is None or len(current_values) == 0 or len(reference_values) == 0:
            # For now, we'll use a simple approach - compare to a baseline
            # In practice, you'd want to store historical baselines
            return {"drift_detected": False, "ks_statistic": 0, "p_value": 1.0}

        # Simple statistical drift detection using KS test
        try:
            from scipy import stats

            ks_stat, p_value = stats.ks_2samp(reference_values, current_values)

            # If p-value is low, it suggests significant difference (drift)
            drift_detected = p_value < 0.05

            return {"drift_detected": drift_detected, "ks_statistic": ks_stat, "p_value": p_value}
        except ImportError:
            # If scipy is not available, return a simple result
            return {"drift_detected": False, "ks_statistic": 0, "p_value": 1.0}

    @staticmethod
    def calculate_psi(reference_values: List[float], current_values: List[float], bins: int = 10) -> float:
        """
        Calculate Population Stability Index.

        PSI > 0.2 is treated as material drift for retraining decisions.
        """
        reference = np.array([v for v in reference_values if v is not None], dtype=float)
        current = np.array([v for v in current_values if v is not None], dtype=float)

        if len(reference) == 0 or len(current) == 0:
            return 0.0

        min_value = min(reference.min(), current.min())
        max_value = max(reference.max(), current.max())
        if min_value == max_value:
            return 0.0

        bin_edges = np.linspace(min_value, max_value, bins + 1)
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        epsilon = 1e-6
        ref_pct = np.maximum(ref_counts / max(len(reference), 1), epsilon)
        cur_pct = np.maximum(cur_counts / max(len(current), 1), epsilon)
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return round(float(psi), 6)

    def detect_feature_drift_psi(
        self,
        feature_name: str,
        current_values: List[float],
        reference_values: List[float],
        threshold: Optional[float] = None,
    ):
        threshold = self.PSI_RETRAIN_THRESHOLD if threshold is None else threshold
        psi_value = self.calculate_psi(reference_values, current_values)
        return {
            "feature": feature_name,
            "psi": psi_value,
            "threshold": threshold,
            "drift_detected": psi_value > threshold,
            "trigger_retraining": psi_value > threshold,
        }

    @staticmethod
    def page_hinkley_test(
        scores: List[float],
        delta: float = PAGE_HINKLEY_DELTA,
        lambda_: float = PAGE_HINKLEY_LAMBDA,
    ):
        """Detect concept drift in model scores using the Page-Hinkley test."""
        clean_scores = [float(score) for score in scores if score is not None]
        if len(clean_scores) < 2:
            return {
                "drift_detected": False,
                "max_statistic": 0.0,
                "threshold": lambda_,
                "mean_score": clean_scores[0] if clean_scores else 0.0,
            }

        running_mean = 0.0
        cumulative = 0.0
        min_cumulative = 0.0
        max_statistic = 0.0

        for index, score in enumerate(clean_scores, 1):
            running_mean += (score - running_mean) / index
            cumulative += score - running_mean - delta
            min_cumulative = min(min_cumulative, cumulative)
            max_statistic = max(max_statistic, cumulative - min_cumulative)

        return {
            "drift_detected": max_statistic > lambda_,
            "max_statistic": round(float(max_statistic), 6),
            "threshold": lambda_,
            "mean_score": round(float(running_mean), 6),
        }

    def evaluate_drift_retraining_trigger(
        self,
        *,
        psi_results: Optional[List[dict]] = None,
        score_drift_result: Optional[dict] = None,
        audit_log_path: Optional[str] = None,
        actor: str = "drift-monitor",
    ):
        """Return a retraining decision and audit the trigger condition when drift is material."""
        psi_results = psi_results or []
        score_drift_result = score_drift_result or {}

        drifted_features = [
            result for result in psi_results if result.get("trigger_retraining") or result.get("drift_detected")
        ]
        score_drifted = bool(score_drift_result.get("drift_detected"))
        trigger = bool(drifted_features or score_drifted)

        decision = {
            "trigger_retraining": trigger,
            "trigger_reason": None,
            "drifted_features": drifted_features,
            "score_drift": score_drift_result,
        }

        if not trigger:
            return decision

        if drifted_features:
            top_feature = max(drifted_features, key=lambda item: item.get("psi", 0.0))
            decision["trigger_reason"] = f"PSI>{top_feature.get('threshold')}:{top_feature.get('feature')}"
        else:
            decision["trigger_reason"] = "PAGE_HINKLEY_SCORE_DRIFT"

        append_change_audit_event(
            {
                "event_type": "RETRAINING_TRIGGER",
                "actor": actor,
                "entity_id": "fraud_model",
                "action": "TRIGGER_RETRAINING",
                "trigger_reason": decision["trigger_reason"],
                "drifted_features": drifted_features,
                "score_drift": score_drift_result,
            },
            audit_log_path=audit_log_path,
        )
        return decision

    def get_drift_retraining_assessment(
        self,
        features: Optional[List[str]] = None,
        current_period_days: int = 7,
        baseline_period_days: int = 30,
        audit: bool = False,
        audit_log_path: Optional[str] = None,
    ):
        """Assess PSI/concept drift from recorded alerts and optionally audit a retraining trigger."""
        features = features or ["amount_tnd", "ml_probability"]
        psi_results = []

        for feature in features:
            comparison = self.get_distribution_comparison(
                feature,
                current_period_days=current_period_days,
                baseline_period_days=baseline_period_days,
            )
            if comparison.get("error"):
                psi_results.append(
                    {
                        "feature": feature,
                        "error": comparison["error"],
                        "drift_detected": False,
                        "trigger_retraining": False,
                    }
                )
                continue

            psi_results.append(
                self.detect_feature_drift_psi(
                    feature,
                    current_values=comparison.get("current_values", []),
                    reference_values=comparison.get("baseline_values", []),
                )
            )

        score_values = []
        score_comparison = self.get_distribution_comparison(
            "ml_probability",
            current_period_days=current_period_days,
            baseline_period_days=baseline_period_days,
        )
        if not score_comparison.get("error"):
            score_values = score_comparison.get("baseline_values", []) + score_comparison.get("current_values", [])

        score_drift = self.page_hinkley_test(score_values)
        decision = (
            self.evaluate_drift_retraining_trigger(
                psi_results=psi_results,
                score_drift_result=score_drift,
                audit_log_path=audit_log_path,
            )
            if audit
            else {
                "trigger_retraining": any(result.get("trigger_retraining") for result in psi_results)
                or bool(score_drift.get("drift_detected")),
                "trigger_reason": None,
                "drifted_features": [
                    result for result in psi_results if result.get("trigger_retraining") or result.get("drift_detected")
                ],
                "score_drift": score_drift,
            }
        )

        return {
            "psi_results": psi_results,
            "score_drift": score_drift,
            "decision": decision,
            "current_period_days": current_period_days,
            "baseline_period_days": baseline_period_days,
            "basis": "Recorded alert feature distributions and model score stream.",
        }

    # Columns allowed for distribution comparison queries (whitelist)
    ALLOWED_DISTRIBUTION_COLUMNS = {"amount_tnd", "ml_probability"}

    def get_distribution_comparison(
        self, feature_name: str, current_period_days: int = 7, baseline_period_days: int = 30
    ):
        """Compare current feature distribution to baseline"""
        try:
            # Whitelist check to prevent SQL injection via column name
            if feature_name not in self.ALLOWED_DISTRIBUTION_COLUMNS:
                return {
                    "current_values": [],
                    "baseline_values": [],
                    "current_mean": 0,
                    "baseline_mean": 0,
                    "current_median": 0,
                    "baseline_median": 0,
                    "error": f"Column '{feature_name}' is not allowed for distribution comparison",
                }

            conn = get_sqlite_connection(str(self.db_path))
            cursor = conn.cursor()

            # Get current period data
            current_start = (datetime.now() - timedelta(days=current_period_days)).strftime("%Y-%m-%d")
            cursor.execute(
                f"""
                SELECT {feature_name}
                FROM high_risk_alerts
                WHERE timestamp >= ?
                AND {feature_name} IS NOT NULL
            """,
                (current_start,),
            )
            current_values = [row[0] for row in cursor.fetchall()]

            # Get baseline period data
            baseline_end = (datetime.now() - timedelta(days=current_period_days)).strftime("%Y-%m-%d")
            baseline_start = (datetime.now() - timedelta(days=baseline_period_days)).strftime("%Y-%m-%d")
            cursor.execute(
                f"""
                SELECT {feature_name}
                FROM high_risk_alerts
                WHERE timestamp BETWEEN ? AND ?
                AND {feature_name} IS NOT NULL
            """,
                (baseline_start, baseline_end),
            )
            baseline_values = [row[0] for row in cursor.fetchall()]

            conn.close()

            return {
                "current_values": current_values,
                "baseline_values": baseline_values,
                "current_mean": statistics.mean(current_values) if current_values else 0,
                "baseline_mean": statistics.mean(baseline_values) if baseline_values else 0,
                "current_median": statistics.median(current_values) if current_values else 0,
                "baseline_median": statistics.median(baseline_values) if baseline_values else 0,
            }
        except Exception as e:
            print(f"Error getting distribution comparison: {e}")
            return {
                "current_values": [],
                "baseline_values": [],
                "current_mean": 0,
                "baseline_mean": 0,
                "current_median": 0,
                "baseline_median": 0,
            }
