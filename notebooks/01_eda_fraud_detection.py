#!/usr/bin/env python3
"""
Exploratory Data Analysis for Tunisian Fraud Detection System
=============================================================

IMPORTANT: This is EDA ONLY. No feature engineering, scaling, or model training
parameters should be derived from this analysis. All ML pipeline parameters
must be computed exclusively on training data after the train/test split.

See src/ml/train_pipeline.py for the proper ML pipeline with NO data leakage.

Usage:
    python notebooks/01_eda_fraud_detection.py

Sections:
    1. Transaction distribution analysis
    2. Fraud pattern visualization
    3. Geographic risk heatmap
    4. Payment method risk profiles
    5. D17/Flouci smurfing pattern analysis
    6. Model performance review (descriptive only, no train/test contamination)
"""

import sys
import os
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

# Optional: only import plotting if available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Visualization will be text-only.")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# 1. Data Loading
# ============================================================================

def load_alerts_data(db_path=None):
    """Load alerts data from the feedback database."""
    if db_path is None:
        db_path = PROJECT_ROOT / "data" / "feedback.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Run the system first to generate data, or use: python test_end_to_end.py")
        return pd.DataFrame(), pd.DataFrame()

    conn = sqlite3.connect(str(db_path))

    alerts_df = pd.read_sql_query("""
        SELECT transaction_id, user_id, amount_tnd, governorate, payment_method,
               branch_id, timestamp, ml_probability, alert_type, created_at
        FROM high_risk_alerts
        ORDER BY created_at DESC
    """, conn)

    feedback_df = pd.read_sql_query("""
        SELECT fl.transaction_id, fl.analyst_label, fl.analyst_comment,
               fl.branch_id, fl.timestamp as feedback_timestamp,
               hra.amount_tnd, hra.governorate, hra.payment_method,
               hra.ml_probability, hra.alert_type
        FROM feedback_labels fl
        LEFT JOIN high_risk_alerts hra ON fl.transaction_id = hra.transaction_id
        WHERE fl.analyst_label IS NOT NULL
    """, conn)

    conn.close()
    return alerts_df, feedback_df


def load_parquet_data():
    """Load silver layer parquet data if available."""
    parquet_path = PROJECT_ROOT / "data" / "parquet" / "silver_fraud_alerts"
    if not parquet_path.exists():
        print(f"Parquet data not found at {parquet_path}")
        return pd.DataFrame()

    try:
        return pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading parquet data: {e}")
        return pd.DataFrame()


# ============================================================================
# 2. Transaction Distribution Analysis
# ============================================================================

def analyze_transaction_distribution(alerts_df):
    """Analyze the distribution of transaction amounts."""
    print("\n" + "=" * 60)
    print("TRANSACTION DISTRIBUTION ANALYSIS")
    print("=" * 60)

    if alerts_df.empty:
        print("No data available.")
        return

    print(f"\nTotal alerts: {len(alerts_df)}")
    print(f"\nAmount (TND) statistics:")
    print(alerts_df["amount_tnd"].describe().to_string())

    # Alert type breakdown
    print(f"\nAlert type breakdown:")
    type_counts = alerts_df["alert_type"].value_counts()
    for alert_type, count in type_counts.items():
        print(f"  {alert_type}: {count} ({count/len(alerts_df)*100:.1f}%)")

    if HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Amount Distribution", "Alert Types"))

        fig.add_trace(
            go.Histogram(x=alerts_df["amount_tnd"], nbinsx=50, name="Amount (TND)"),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=type_counts.index, y=type_counts.values, name="Alert Type"),
            row=1, col=2
        )

        fig.update_layout(title="Transaction Distribution Overview", showlegend=False)
        fig.write_html(str(PROJECT_ROOT / "notebooks" / "output_distribution.html"))
        print("\nPlot saved to notebooks/output_distribution.html")


# ============================================================================
# 3. Geographic Risk Analysis
# ============================================================================

def analyze_geographic_risk(alerts_df):
    """Analyze fraud patterns by governorate."""
    print("\n" + "=" * 60)
    print("GEOGRAPHIC RISK ANALYSIS")
    print("=" * 60)

    if alerts_df.empty:
        print("No data available.")
        return

    gov_stats = alerts_df.groupby("governorate").agg(
        total_alerts=("transaction_id", "count"),
        avg_amount=("amount_tnd", "mean"),
        avg_ml_prob=("ml_probability", "mean"),
        max_amount=("amount_tnd", "max"),
    ).sort_values("total_alerts", ascending=False)

    print(f"\nAlerts by Governorate:")
    print(gov_stats.to_string())

    if HAS_PLOTLY:
        fig = px.bar(
            gov_stats.reset_index(),
            x="governorate", y="total_alerts",
            color="avg_ml_prob",
            color_continuous_scale="Reds",
            title="Fraud Alerts by Governorate (colored by avg ML probability)"
        )
        fig.write_html(str(PROJECT_ROOT / "notebooks" / "output_geographic.html"))
        print("\nPlot saved to notebooks/output_geographic.html")


# ============================================================================
# 4. Payment Method Risk Profiles
# ============================================================================

def analyze_payment_methods(alerts_df, feedback_df):
    """Analyze risk profiles by payment method."""
    print("\n" + "=" * 60)
    print("PAYMENT METHOD RISK PROFILES")
    print("=" * 60)

    if alerts_df.empty:
        print("No data available.")
        return

    method_stats = alerts_df.groupby("payment_method").agg(
        total_alerts=("transaction_id", "count"),
        avg_amount=("amount_tnd", "mean"),
        avg_ml_prob=("ml_probability", "mean"),
    ).sort_values("avg_ml_prob", ascending=False)

    print(f"\nRisk by Payment Method:")
    print(method_stats.to_string())

    # If feedback available, show confirmed fraud rate per method
    if not feedback_df.empty:
        method_feedback = feedback_df.groupby("payment_method").apply(
            lambda g: pd.Series({
                "confirmed_fraud": (g["analyst_label"] == "Confirmed Fraud").sum(),
                "false_positive": (g["analyst_label"] == "False Positive").sum(),
                "precision": (g["analyst_label"] == "Confirmed Fraud").mean()
            })
        ).sort_values("precision", ascending=False)

        print(f"\nConfirmed Fraud Rate by Payment Method:")
        print(method_feedback.to_string())


# ============================================================================
# 5. D17/Flouci Smurfing Pattern Analysis
# ============================================================================

def analyze_smurfing_patterns(alerts_df):
    """Analyze potential structuring/smurfing patterns."""
    print("\n" + "=" * 60)
    print("D17/FLOUCI SMURFING PATTERN ANALYSIS")
    print("=" * 60)

    if alerts_df.empty:
        print("No data available.")
        return

    # Transactions in the smurfing range (1400-1500 TND)
    smurfing_range = alerts_df[
        (alerts_df["amount_tnd"] >= 1400) &
        (alerts_df["amount_tnd"] <= 1500)
    ]
    print(f"\nTransactions in smurfing range (1400-1500 TND): {len(smurfing_range)}")
    print(f"  As percentage of total: {len(smurfing_range)/max(len(alerts_df),1)*100:.2f}%")

    # Flouci-specific analysis
    flouci = alerts_df[alerts_df["payment_method"] == "Flouci"]
    print(f"\nFlouci transactions: {len(flouci)}")
    if not flouci.empty:
        print(f"  Average amount: {flouci['amount_tnd'].mean():.2f} TND")
        print(f"  Average ML probability: {flouci['ml_probability'].mean():.3f}")

    # High-value Flouci (D17 rule trigger: >2000 TND)
    flouci_high = flouci[flouci["amount_tnd"] > 2000]
    print(f"  Flouci > 2000 TND (D17 trigger): {len(flouci_high)}")


# ============================================================================
# 6. Model Performance Review
# ============================================================================

def analyze_model_performance(feedback_df):
    """Analyze model performance based on analyst feedback."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE REVIEW")
    print("=" * 60)

    if feedback_df.empty:
        print("No feedback data available.")
        return

    print(f"\nTotal feedback records: {len(feedback_df)}")

    # Overall label distribution
    label_counts = feedback_df["analyst_label"].value_counts()
    print(f"\nFeedback distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(feedback_df)*100:.1f}%)")

    # Performance by alert type
    for alert_type in feedback_df["alert_type"].unique():
        subset = feedback_df[feedback_df["alert_type"] == alert_type]
        if not subset.empty:
            confirmed = (subset["analyst_label"] == "Confirmed Fraud").sum()
            total = len(subset)
            print(f"\n  {alert_type}: {confirmed}/{total} confirmed fraud ({confirmed/total*100:.1f}%)")

    # ML probability calibration
    if "ml_probability" in feedback_df.columns:
        fraud_probs = feedback_df[feedback_df["analyst_label"] == "Confirmed Fraud"]["ml_probability"]
        fp_probs = feedback_df[feedback_df["analyst_label"] == "False Positive"]["ml_probability"]

        if not fraud_probs.empty:
            print(f"\nML probability for Confirmed Fraud:")
            print(f"  Mean: {fraud_probs.mean():.3f}, Median: {fraud_probs.median():.3f}")
        if not fp_probs.empty:
            print(f"ML probability for False Positives:")
            print(f"  Mean: {fp_probs.mean():.3f}, Median: {fp_probs.median():.3f}")

        if HAS_PLOTLY:
            fig = go.Figure()
            if not fraud_probs.empty:
                fig.add_trace(go.Histogram(x=fraud_probs, name="Confirmed Fraud", opacity=0.7))
            if not fp_probs.empty:
                fig.add_trace(go.Histogram(x=fp_probs, name="False Positive", opacity=0.7))
            fig.update_layout(
                title="ML Probability Distribution by Analyst Label",
                xaxis_title="ML Probability",
                yaxis_title="Count",
                barmode="overlay"
            )
            fig.write_html(str(PROJECT_ROOT / "notebooks" / "output_ml_calibration.html"))
            print("\nPlot saved to notebooks/output_ml_calibration.html")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("TUNISIAN FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    alerts_df, feedback_df = load_alerts_data()
    silver_df = load_parquet_data()

    if not alerts_df.empty:
        analyze_transaction_distribution(alerts_df)
        analyze_geographic_risk(alerts_df)
        analyze_payment_methods(alerts_df, feedback_df)
        analyze_smurfing_patterns(alerts_df)

    if not feedback_df.empty:
        analyze_model_performance(feedback_df)

    if not silver_df.empty:
        print("\n" + "=" * 60)
        print(f"SILVER LAYER: {len(silver_df)} records available")
        print("=" * 60)
        print(silver_df.describe().to_string())

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
