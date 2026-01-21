# dashboard/monitoring.py
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import statistics
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared.utils import get_sqlite_connection

class ForensicAnalyticEngine:
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
            return {
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "total_calls": 0
            }
        
        latencies = list(self.inference_latencies)
        return {
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "p99_latency_ms": round(np.percentile(latencies, 99), 2),
            "total_calls": len(latencies)
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
                "prob_label_pairs": prob_label_pairs
            }
        except Exception as e:
            print(f"Error getting feedback analysis: {e}")
            return {
                "precision": 0,
                "feedback_counts": {},
                "total_feedback": 0,
                "prob_label_pairs": []
            }

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
            recall = tp / sum(1 for _, label in prob_label_pairs if label == "Confirmed Fraud") if sum(1 for _, label in prob_label_pairs if label == "Confirmed Fraud") > 0 else 0

            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            threshold_analysis[threshold] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1_score, 3)
            }

        # Find threshold with highest F1 score
        best_threshold = max(thresholds, key=lambda t: threshold_analysis[t]["f1_score"])

        return {
            "optimal_threshold": best_threshold,
            "threshold_analysis": threshold_analysis
        }

    def detect_feature_drift(self, feature_name: str, current_values: List[float], reference_values: List[float] = None):
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

            return {
                "drift_detected": drift_detected,
                "ks_statistic": ks_stat,
                "p_value": p_value
            }
        except ImportError:
            # If scipy is not available, return a simple result
            return {"drift_detected": False, "ks_statistic": 0, "p_value": 1.0}

    def get_distribution_comparison(self, feature_name: str, current_period_days: int = 7, baseline_period_days: int = 30):
        """Compare current feature distribution to baseline"""
        try:
            conn = get_sqlite_connection(str(self.db_path))
            cursor = conn.cursor()

            # Get current period data
            current_start = (datetime.now() - timedelta(days=current_period_days)).strftime('%Y-%m-%d')
            cursor.execute(f"""
                SELECT {feature_name}
                FROM high_risk_alerts
                WHERE timestamp >= '{current_start}'
                AND {feature_name} IS NOT NULL
            """)
            current_values = [row[0] for row in cursor.fetchall()]

            # Get baseline period data
            baseline_end = (datetime.now() - timedelta(days=current_period_days)).strftime('%Y-%m-%d')
            baseline_start = (datetime.now() - timedelta(days=baseline_period_days)).strftime('%Y-%m-%d')
            cursor.execute(f"""
                SELECT {feature_name}
                FROM high_risk_alerts
                WHERE timestamp BETWEEN '{baseline_start}' AND '{baseline_end}'
                AND {feature_name} IS NOT NULL
            """)
            baseline_values = [row[0] for row in cursor.fetchall()]

            conn.close()

            return {
                "current_values": current_values,
                "baseline_values": baseline_values,
                "current_mean": statistics.mean(current_values) if current_values else 0,
                "baseline_mean": statistics.mean(baseline_values) if baseline_values else 0,
                "current_median": statistics.median(current_values) if current_values else 0,
                "baseline_median": statistics.median(baseline_values) if baseline_values else 0
            }
        except Exception as e:
            print(f"Error getting distribution comparison: {e}")
            return {
                "current_values": [],
                "baseline_values": [],
                "current_mean": 0,
                "baseline_mean": 0,
                "current_median": 0,
                "baseline_median": 0
            }
