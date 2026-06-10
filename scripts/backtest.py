#!/usr/bin/env python3
"""
Backtesting Framework for Amastan Fraud Shield Guard

Replays historical transaction data through the fraud detection pipeline with
modified rule weights or model versions to measure the impact on false positives,
true positives, and alert volume.

This prevents "tuning in the dark" — every rule change is validated against
historical data before deployment.

Usage:
    python scripts/backtest.py                              # Default backtest
    python scripts/backtest.py --rule velocity --weight 0.4 # Change velocity weight
    python backtest.py --date-from 2026-01-01 --date-to 2026-02-01
    python backtest.py --model-path models/registry/new_model --output report.json
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.shared.risk_config import (
    ALERT_SCORE_THRESHOLD,
    EWALLET_REVIEW_THRESHOLD_TND,
    HIGH_VALUE_REVIEW_THRESHOLD_TND,
    PROXY_LABEL_THRESHOLD,
    RISK_WEIGHTS,
    TRAVEL_REVIEW_THRESHOLD,
    VELOCITY_REVIEW_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    test_id: str = ""
    run_at: str = ""
    data_from: str = ""
    data_to: str = ""
    total_transactions: int = 0
    total_fraud: int = 0
    total_legitimate: int = 0

    # Original pipeline results
    original_tp: int = 0
    original_fp: int = 0
    original_tn: int = 0
    original_fn: int = 0
    original_precision: float = 0.0
    original_recall: float = 0.0
    original_f1: float = 0.0
    original_alert_count: int = 0

    # Modified pipeline results
    modified_tp: int = 0
    modified_fp: int = 0
    modified_tn: int = 0
    modified_fn: int = 0
    modified_precision: float = 0.0
    modified_recall: float = 0.0
    modified_f1: float = 0.0
    modified_alert_count: int = 0

    # Delta (change)
    delta_fp: int = 0
    delta_tp: int = 0
    delta_precision: float = 0.0
    delta_recall: float = 0.0
    delta_f1: float = 0.0
    delta_alert_count: int = 0

    # Recommendation
    recommendation: str = ""
    label_source: str = ""
    changes_applied: dict = None


class BacktestEngine:
    """
    Replays historical data through the fraud detection pipeline.

    The engine:
    1. Loads historical parquet data
    2. Runs it through the ORIGINAL pipeline (current rules)
    3. Runs it through the MODIFIED pipeline (changed rules/model)
    4. Compares results
    """

    def __init__(self, parquet_path: str = "./data/parquet/silver_fraud_alerts"):
        self.parquet_path = parquet_path

    def load_data(self, date_from: str = None, date_to: str = None) -> pd.DataFrame:
        """
        Load historical parquet data with optional date filtering.

        Args:
            date_from: ISO date string (e.g., "2026-01-01")
            date_to: ISO date string

        Returns:
            DataFrame of historical transactions.
        """
        path = Path(self.parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet data not found: {self.parquet_path}")

        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} transactions from {path}")

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            if date_from:
                df = df[df["timestamp"] >= pd.to_datetime(date_from)]
                logger.info(f"Filtered from {date_from}: {len(df):,} transactions")
            if date_to:
                df = df[df["timestamp"] <= pd.to_datetime(date_to)]
                logger.info(f"Filtered to {date_to}: {len(df):,} transactions")

        return df

    @staticmethod
    def _apply_rules(
        df: pd.DataFrame,
        score_prefix: str,
        weights: dict,
        thresholds: dict,
        alert_threshold: float,
    ) -> pd.DataFrame:
        """Apply one rule configuration without duplicating scoring logic."""
        result = df.copy()
        v_count = result.get("v_count", pd.Series(0, index=result.index))
        g_dist = result.get("g_dist", pd.Series(0, index=result.index))
        payment_method = result.get("payment_method", pd.Series("", index=result.index))

        result["velocity_risk"] = (v_count > thresholds["velocity"]).astype(float)
        result["travel_risk"] = (g_dist > thresholds["travel"]).astype(float)
        result["high_value_risk"] = (result["amount_tnd"] > thresholds["high_value"]).astype(float)
        result["d17_risk"] = (
            (result["amount_tnd"] > thresholds["ewallet"]) & payment_method.fillna("").str.lower().eq("flouci")
        ).astype(float)

        risk_score_col = f"{score_prefix}_risk_score"
        score_col = f"{score_prefix}_score"
        alert_col = f"{score_prefix}_alert"
        result[risk_score_col] = (
            result["velocity_risk"] * weights["velocity"]
            + result["travel_risk"] * weights["travel"]
            + result["high_value_risk"] * weights["high_value"]
            + result["d17_risk"] * weights["d17_limit"]
        )
        result[score_col] = result[risk_score_col]
        result[alert_col] = result[score_col] > alert_threshold
        return result

    def _apply_original_rules(
        self,
        df: pd.DataFrame,
        alert_threshold: float = ALERT_SCORE_THRESHOLD,
    ) -> pd.DataFrame:
        """Apply the current prototype configuration."""
        result = self._apply_rules(
            df,
            "original",
            RISK_WEIGHTS.copy(),
            self._default_thresholds(),
            alert_threshold,
        )
        if "ml_probability" in result.columns:
            result["original_score"] = result["ml_probability"]
            result["original_alert"] = result["original_score"] > alert_threshold
        return result

    @staticmethod
    def _default_thresholds() -> dict:
        return {
            "velocity": VELOCITY_REVIEW_THRESHOLD,
            "travel": TRAVEL_REVIEW_THRESHOLD,
            "high_value": HIGH_VALUE_REVIEW_THRESHOLD_TND,
            "ewallet": EWALLET_REVIEW_THRESHOLD_TND,
        }

    def _apply_modified_rules(
        self,
        df: pd.DataFrame,
        weight_changes: dict = None,
        threshold_changes: dict = None,
        new_model_path: str = None,
        alert_threshold: float = ALERT_SCORE_THRESHOLD,
    ) -> pd.DataFrame:
        """Apply modified rules to the data."""
        weights = RISK_WEIGHTS.copy()
        if weight_changes:
            weights.update(weight_changes)
            logger.info(f"Modified weights: {weights}")

        thresholds = self._default_thresholds()
        if threshold_changes:
            aliases = {
                "velocity_threshold": "velocity",
                "travel_threshold": "travel",
                "high_value_threshold": "high_value",
                "d17_threshold": "ewallet",
                "ewallet_threshold": "ewallet",
            }
            for external_name, value in threshold_changes.items():
                key = aliases.get(external_name, external_name)
                if key in thresholds:
                    thresholds[key] = value
            logger.info(f"Modified thresholds: {thresholds}")

        result = self._apply_rules(df, "modified", weights, thresholds, alert_threshold)

        if new_model_path and Path(new_model_path).exists():
            try:
                import joblib

                model_path = Path(new_model_path)
                artifact_path = model_path / "pipeline.pkl" if model_path.is_dir() else model_path
                model_data = joblib.load(artifact_path)
                model = model_data.get("model") if isinstance(model_data, dict) else model_data

                feature_cols = (
                    model_data.get("feature_columns", ["amount_tnd", "v_count", "g_dist"])
                    if isinstance(model_data, dict)
                    else getattr(model, "feature_names_in_", None)
                )
                if feature_cols:
                    available = [c for c in feature_cols if c in result.columns]
                    if len(available) == len(feature_cols):
                        predictions = model.predict_proba(result[available])[:, 1]
                        result["modified_score"] = predictions
                        logger.info(f"Scored with new model: {artifact_path}")
            except Exception as e:
                logger.warning(f"New model scoring failed: {e}, using rule-based score")

        result["modified_alert"] = result["modified_score"] > alert_threshold
        return result

    def _compute_metrics(self, df: pd.DataFrame, score_col: str, alert_col: str) -> dict:
        """Compute precision, recall, F1 from scored data."""
        label_source = "verified"
        if "label" not in df.columns:
            if "ml_probability" in df.columns:
                df = df.copy()
                df["label"] = (df["ml_probability"] > PROXY_LABEL_THRESHOLD).astype(int)
                label_source = "ml_probability_proxy"
            else:
                raise ValueError("Backtesting requires a verified label or ml_probability proxy")

        tp = ((df[alert_col]) & (df["label"] == 1)).sum()
        fp = ((df[alert_col]) & (df["label"] == 0)).sum()
        tn = ((~df[alert_col]) & (df["label"] == 0)).sum()
        fn = ((~df[alert_col]) & (df["label"] == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "alert_count": int(df[alert_col].sum()),
            "label_source": label_source,
        }

    def run(
        self,
        weight_changes: dict = None,
        threshold_changes: dict = None,
        new_model_path: str = None,
        date_from: str = None,
        date_to: str = None,
        alert_threshold: float = ALERT_SCORE_THRESHOLD,
    ) -> BacktestResult:
        """
        Run the full backtest: original vs modified.

        Args:
            weight_changes: Dict of rule name -> new weight
            threshold_changes: Dict of rule name -> new threshold
            new_model_path: Path to a new model file
            date_from: Start date for data filtering
            date_to: End date for data filtering
            alert_threshold: Score threshold for alert generation

        Returns:
            BacktestResult with comparison metrics.
        """
        result = BacktestResult()
        result.test_id = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
        result.run_at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        result.changes_applied = {
            "weight_changes": weight_changes or {},
            "threshold_changes": threshold_changes or {},
            "new_model_path": new_model_path,
            "alert_threshold": alert_threshold,
        }

        # Load data
        df = self.load_data(date_from, date_to)
        if df.empty:
            raise ValueError("No data available for backtesting")

        result.data_from = date_from or "earliest"
        result.data_to = date_to or "latest"
        result.total_transactions = len(df)
        labels = df["label"] if "label" in df.columns else (df["ml_probability"] > PROXY_LABEL_THRESHOLD).astype(int)
        result.total_fraud = int(labels.sum())
        result.total_legitimate = result.total_transactions - result.total_fraud

        # Apply original rules
        logger.info("Applying original rules...")
        original_df = self._apply_original_rules(df, alert_threshold=alert_threshold)
        original_metrics = self._compute_metrics(original_df, "original_score", "original_alert")

        result.original_tp = original_metrics["tp"]
        result.original_fp = original_metrics["fp"]
        result.original_tn = original_metrics["tn"]
        result.original_fn = original_metrics["fn"]
        result.original_precision = original_metrics["precision"]
        result.original_recall = original_metrics["recall"]
        result.original_f1 = original_metrics["f1"]
        result.original_alert_count = original_metrics["alert_count"]

        # Apply modified rules
        logger.info("Applying modified rules...")
        modified_df = self._apply_modified_rules(
            df,
            weight_changes=weight_changes,
            threshold_changes=threshold_changes,
            new_model_path=new_model_path,
            alert_threshold=alert_threshold,
        )
        modified_metrics = self._compute_metrics(modified_df, "modified_score", "modified_alert")

        result.modified_tp = modified_metrics["tp"]
        result.modified_fp = modified_metrics["fp"]
        result.modified_tn = modified_metrics["tn"]
        result.modified_fn = modified_metrics["fn"]
        result.modified_precision = modified_metrics["precision"]
        result.modified_recall = modified_metrics["recall"]
        result.modified_f1 = modified_metrics["f1"]
        result.modified_alert_count = modified_metrics["alert_count"]
        result.label_source = original_metrics["label_source"]

        # Calculate deltas
        result.delta_tp = result.modified_tp - result.original_tp
        result.delta_fp = result.modified_fp - result.original_fp
        result.delta_precision = result.modified_precision - result.original_precision
        result.delta_recall = result.modified_recall - result.original_recall
        result.delta_f1 = result.modified_f1 - result.original_f1
        result.delta_alert_count = result.modified_alert_count - result.original_alert_count

        # Generate recommendation
        result.recommendation = self._generate_recommendation(result)

        return result

    @staticmethod
    def _generate_recommendation(r: BacktestResult) -> str:
        """Generate a deployment recommendation based on backtest results."""
        if r.label_source != "verified":
            return "NON-DECISIONAL: Proxy labels were used; collect reviewed labels before deployment."

        issues = []

        if r.delta_fp > r.original_fp * 0.1:
            issues.append(f"False positives increased by {r.delta_fp} ({r.delta_fp/max(r.original_fp,1)*100:.1f}%)")

        if r.delta_f1 < -0.05:
            issues.append(f"F1 score degraded by {r.delta_f1:.4f}")

        if r.delta_recall < -0.1:
            issues.append(f"Recall dropped by {r.delta_recall:.4f} (missed frauds)")

        if r.delta_alert_count > r.original_alert_count * 0.5:
            issues.append(
                f"Alert volume increased by {r.delta_alert_count} ({r.delta_alert_count/max(r.original_alert_count,1)*100:.1f}%) — may overwhelm analysts"
            )

        if issues:
            return f"DO NOT DEPLOY: {'; '.join(issues)}"

        if r.delta_f1 > 0.02:
            return f"DEPLOY: F1 improved by {r.delta_f1:.4f} (from {r.original_f1:.4f} to {r.modified_f1:.4f})"
        elif r.delta_fp < 0 and r.delta_recall >= 0:
            return f"DEPLOY: Reduced false positives by {-r.delta_fp} without losing recall"
        else:
            return f"NEUTRAL: No significant improvement. Changes may not be worth the operational risk."

    def print_report(self, result: BacktestResult):
        """Print a formatted backtest report."""
        print("\n" + "=" * 80)
        print("  AMASTAN BACKTEST REPORT")
        print("=" * 80)
        print(f"\n  Test ID:       {result.test_id}")
        print(f"  Run at:        {result.run_at}")
        print(f"  Data range:    {result.data_from} to {result.data_to}")
        print(
            f"  Transactions:  {result.total_transactions:,} ({result.total_fraud} fraud, {result.total_legitimate} legitimate)"
        )

        if result.changes_applied.get("weight_changes"):
            print(f"\n  Weight changes: {result.changes_applied['weight_changes']}")
        if result.changes_applied.get("threshold_changes"):
            print(f"  Threshold changes: {result.changes_applied['threshold_changes']}")
        if result.changes_applied.get("new_model_path"):
            print(f"  New model: {result.changes_applied['new_model_path']}")

        print("\n" + "-" * 80)
        print(f"  {'METRIC':<25} {'ORIGINAL':>12} {'MODIFIED':>12} {'DELTA':>12}")
        print("-" * 80)
        print(f"  {'True Positives':<25} {result.original_tp:>12} {result.modified_tp:>12} {result.delta_tp:>+12}")
        print(f"  {'False Positives':<25} {result.original_fp:>12} {result.modified_fp:>12} {result.delta_fp:>+12}")
        print(f"  {'True Negatives':<25} {result.original_tn:>12} {result.modified_tn:>12} {'':>12}")
        print(f"  {'False Negatives':<25} {result.original_fn:>12} {result.modified_fn:>12} {'':>12}")
        print("-" * 80)
        print(
            f"  {'Precision':<25} {result.original_precision:>12.4f} {result.modified_precision:>12.4f} {result.delta_precision:>+12.4f}"
        )
        print(
            f"  {'Recall':<25} {result.original_recall:>12.4f} {result.modified_recall:>12.4f} {result.delta_recall:>+12.4f}"
        )
        print(f"  {'F1 Score':<25} {result.original_f1:>12.4f} {result.modified_f1:>12.4f} {result.delta_f1:>+12.4f}")
        print(
            f"  {'Alert Count':<25} {result.original_alert_count:>12} {result.modified_alert_count:>12} {result.delta_alert_count:>+12}"
        )
        print("=" * 80)
        print(f"\n  RECOMMENDATION: {result.recommendation}")
        print("=" * 80 + "\n")

    def save_report(self, result: BacktestResult, output_path: str):
        """Save backtest report as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        logger.info(f"Report saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest fraud detection rules against historical data")
    parser.add_argument(
        "--parquet-path", type=str, default="./data/parquet/silver_fraud_alerts", help="Path to historical parquet data"
    )
    parser.add_argument("--date-from", type=str, help="Start date (ISO)")
    parser.add_argument("--date-to", type=str, help="End date (ISO)")
    parser.add_argument("--rule", type=str, help="Rule name to modify (e.g., velocity)")
    parser.add_argument("--weight", type=float, help="New weight for the rule")
    parser.add_argument("--threshold", type=float, help="New threshold for the rule")
    parser.add_argument("--model-path", type=str, help="Path to new model for comparison")
    parser.add_argument("--output", type=str, help="Output JSON report path")
    parser.add_argument("--alert-threshold", type=float, default=0.5, help="Alert score threshold")
    args = parser.parse_args()

    engine = BacktestEngine(parquet_path=args.parquet_path)

    weight_changes = {}
    threshold_changes = {}

    if args.rule and args.weight is not None:
        weight_changes[args.rule] = args.weight
    if args.rule and args.threshold is not None:
        threshold_changes[f"{args.rule}_threshold"] = args.threshold

    logger.info("Starting backtest...")
    result = engine.run(
        weight_changes=weight_changes or None,
        threshold_changes=threshold_changes or None,
        new_model_path=args.model_path,
        date_from=args.date_from,
        date_to=args.date_to,
        alert_threshold=args.alert_threshold,
    )

    engine.print_report(result)

    if args.output:
        engine.save_report(result, args.output)


if __name__ == "__main__":
    main()
