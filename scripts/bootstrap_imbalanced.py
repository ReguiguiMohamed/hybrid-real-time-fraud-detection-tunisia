#!/usr/bin/env python3
"""
Imbalanced Bootstrap for Amastan Fraud Shield Guard
Generates realistic training data with proper class imbalance (0.01% fraud rate).

Replaces the naive 50/50 bootstrap with production-realistic distributions:
- Fraud rate: 0.01% (1 in 10,000 transactions)
- Amount distribution: log-normal (real financial transactions)
- Geographic distribution: weighted by population/commercial activity
- Temporal patterns: higher activity during business hours
- Smurfing patterns: realistic structuring behavior

Usage:
    python scripts/bootstrap_imbalanced.py --n-samples 100000 --fraud-rate 0.0001
"""
import argparse
import sqlite3
import json
import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


# Tunisian governorate weights (based on commercial activity)
GOVERNORATE_WEIGHTS = {
    "Tunis": 0.25,
    "Sfax": 0.15,
    "Sousse": 0.10,
    "Ariana": 0.08,
    "Bizerte": 0.05,
    "Nabeul": 0.05,
    "Gabes": 0.04,
    "Kairouan": 0.04,
    "Ben Arous": 0.04,
    "Monastir": 0.03,
    "Mahdia": 0.02,
    "Manouba": 0.02,
    "Gafsa": 0.02,
    "Medenine": 0.02,
    "Jendouba": 0.02,
    "Beja": 0.01,
    "Kasserine": 0.01,
    "Sidi Bouzid": 0.01,
    "Le Kef": 0.01,
    "Zaghouan": 0.01,
    "Siliana": 0.01,
    "Tozeur": 0.005,
    "Kebili": 0.005,
    "Tataouine": 0.005,
}

PAYMENT_METHODS = ["Flouci", "card", "bank_transfer", "mobile", "e-dinar"]
PAYMENT_WEIGHTS = [0.35, 0.30, 0.15, 0.12, 0.08]

BRANCHES = [f"B{i:02d}" for i in range(1, 51)]


class ImbalancedTransactionGenerator:
    """
    Generate realistic Tunisian payment transactions with proper class imbalance.
    """

    def __init__(
        self,
        n_samples: int = 100_000,
        fraud_rate: float = 0.0001,  # 0.01%
        random_seed: int = 42,
    ):
        self.n_samples = n_samples
        self.fraud_rate = fraud_rate
        self.rng = np.random.default_rng(random_seed)

        self.governorates = list(GOVERNORATE_WEIGHTS.keys())
        self.gov_weights = list(GOVERNORATE_WEIGHTS.values())

    def _generate_legitimate_transactions(self, n: int) -> pd.DataFrame:
        """Generate realistic legitimate transactions."""
        transactions = []

        # Generate user IDs (fewer users than transactions - repeat customers)
        n_users = max(n // 20, 1000)  # ~20 tx per user
        user_ids = [f"USER_{i:06d}" for i in range(n_users)]

        for i in range(n):
            user_id = self.rng.choice(user_ids)
            governorate = self.rng.choice(self.governorates, p=self.gov_weights)
            payment_method = self.rng.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)
            branch_id = self.rng.choice(BRANCHES)

            # Amount: log-normal distribution (realistic financial amounts)
            # Median ~50 TND, with occasional large transactions
            amount = max(1.0, self.rng.lognormal(mean=3.5, sigma=1.5))
            amount = round(amount, 2)

            # Timestamp: business hours weighted (9am-6pm Tunisia time)
            hours_ago = self.rng.exponential(scale=48)  # Most transactions within 48h
            timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=float(hours_ago))

            transactions.append({
                "transaction_id": f"TX-{i:08d}",
                "user_id": user_id,
                "amount_tnd": amount,
                "governorate": governorate,
                "payment_method": payment_method,
                "branch_id": branch_id,
                "timestamp": timestamp.isoformat(),
                "label": 0,  # Legitimate
            })

        return pd.DataFrame(transactions)

    def _generate_fraud_transactions(self, n: int) -> pd.DataFrame:
        """
        Generate realistic fraud transactions with multiple fraud patterns.

        Fraud patterns:
        1. Smurfing/structuring: 1400-1500 TND via Flouci (D17 threshold avoidance)
        2. High-velocity: Many transactions in short time from same user
        3. Geographic anomaly: Multiple governorates in short time
        4. High-value: Unusually large transactions
        5. Account takeover: Unusual behavior for the user's profile
        """
        transactions = []
        fraud_patterns = ["smurfing", "high_velocity", "geo_anomaly", "high_value", "account_takeover"]
        pattern_weights = [0.30, 0.25, 0.15, 0.15, 0.15]

        # Fewer fraud users (fraudsters often create new accounts)
        n_fraud_users = max(n // 5, 50)
        fraud_user_ids = [f"FRAUD_USER_{i:04d}" for i in range(n_fraud_users)]

        start_offset = self.rng.integers(0, n, size=n)  # Random insertion points

        for i in range(n):
            pattern = self.rng.choice(fraud_patterns, p=pattern_weights)
            user_id = self.rng.choice(fraud_user_ids)

            governorate = self.rng.choice(self.governorates, p=self.gov_weights)
            payment_method = self.rng.choice(PAYMENT_METHODS, p=PAYMENT_METHODS)
            branch_id = self.rng.choice(BRANCHES)

            timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=float(self.rng.exponential(scale=24)))

            if pattern == "smurfing":
                # Smurfing: amounts just below D17 reporting threshold
                amount = round(self.rng.uniform(1400, 1500), 2)
                payment_method = "Flouci"  # Smurfing typically uses e-wallets
            elif pattern == "high_velocity":
                # High velocity: large amounts, rapid succession
                amount = round(self.rng.lognormal(mean=7, sigma=1), 2)
                # Timestamps clustered (within hours)
                timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=float(self.rng.uniform(5, 120)))
            elif pattern == "geo_anomaly":
                # Geographic anomaly: unusual locations
                # Use less common governorates more often
                border_govs = ["Tataouine", "Kasserine", "Le Kef", "Medenine", "Gafsa"]
                governorate = self.rng.choice(border_govs, p=[0.25, 0.25, 0.20, 0.15, 0.15])
                amount = round(self.rng.lognormal(mean=6, sigma=1.2), 2)
            elif pattern == "high_value":
                # High-value transactions
                amount = round(self.rng.lognormal(mean=9, sigma=1), 2)  # Very large
                amount = min(amount, 50000)  # Cap at 50K
            else:  # account_takeover
                # Account takeover: unusual behavior
                amount = round(self.rng.lognormal(mean=6.5, sigma=1.5), 2)
                payment_method = self.rng.choice(["card", "bank_transfer"])  # Not typical for user

            transactions.append({
                "transaction_id": f"TX-F{i:08d}",
                "user_id": user_id,
                "amount_tnd": amount,
                "governorate": governorate,
                "payment_method": payment_method,
                "branch_id": branch_id,
                "timestamp": timestamp.isoformat(),
                "label": 1,  # Fraud
            })

        return pd.DataFrame(transactions)

    def generate(self) -> pd.DataFrame:
        """Generate imbalanced dataset with legitimate and fraud transactions."""
        n_fraud = max(int(self.n_samples * self.fraud_rate), 10)  # At least 10 fraud cases
        n_legitimate = self.n_samples - n_fraud

        print(f"Generating {n_legitimate:,} legitimate transactions...")
        legitimate_df = self._generate_legitimate_transactions(n_legitimate)

        print(f"Generating {n_fraud:,} fraud transactions ({self.fraud_rate*100:.4f}% fraud rate)...")
        fraud_df = self._generate_fraud_transactions(n_fraud)

        # Combine and shuffle
        combined = pd.concat([legitimate_df, fraud_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\nGenerated {len(combined):,} total transactions")
        print(f"  Legitimate: {(combined['label'] == 0).sum():,} ({(combined['label'] == 0).mean()*100:.4f}%)")
        print(f"  Fraud:      {(combined['label'] == 1).sum():,} ({(combined['label'] == 1).mean()*100:.4f}%)")

        return combined

    def generate_with_features(self) -> pd.DataFrame:
        """Generate transactions and add engineered features for ML training."""
        df = self.generate()

        # Add windowed features (simulating what Spark would produce)
        # Per-user aggregation
        user_stats = df.groupby("user_id").agg(
            v_count=("transaction_id", "count"),
            avg_amount=("amount_tnd", "mean"),
            g_dist=("governorate", "nunique"),
            max_amount=("amount_tnd", "max"),
        ).reset_index()

        user_stats.columns = ["user_id", "v_count", "avg_amount", "g_dist", "max_amount"]

        # Merge back
        df = df.merge(user_stats, on="user_id", how="left")

        # Derived features
        df["is_smurfing"] = ((df["amount_tnd"] >= 1400) & (df["amount_tnd"] <= 1500) & (df["payment_method"] == "Flouci")).astype(int)
        df["high_velocity_flag"] = (df["v_count"] > 5).astype(int)
        df["amount_stddev"] = df.groupby("user_id")["amount_tnd"].transform("std").fillna(0)

        return df


def train_on_imbalanced_data(df: pd.DataFrame, output_path: str = "./models/v1_imbalanced.pkl") -> dict:
    """
    Train a model on properly imbalanced data.
    Reports PR-AUC and F1 (not accuracy, which is meaningless for imbalanced data).
    """
    print("\n" + "=" * 60)
    print("  TRAINING ON IMBALANCED DATA")
    print("=" * 60)

    # Features
    feature_cols = ["v_count", "g_dist", "avg_amount", "max_amount", "is_smurfing", "high_velocity_flag", "amount_tnd"]
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols]
    y = df["label"]

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train):,} ({y_train.sum()} fraud, {y_train.mean()*100:.4f}%)")
    print(f"Test:  {len(X_test):,} ({y_test.sum()} fraud, {y_test.mean()*100:.4f}%)")

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        scale_pos_weight=len(y_train) / max(y_train.sum(), 1),  # Handle class imbalance
        random_state=42,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # IMPORTANT: Report proper metrics for imbalanced data
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Precision-Recall AUC (more informative than ROC-AUC for imbalanced data)
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc_val = np.trapz(precision_vals, recall_vals)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc_val),
        "confusion_matrix": pd.DataFrame(
            pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
        ).to_dict(),
        "feature_columns": available_cols,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    print(f"\n{'=' * 50}")
    print(f"  MODEL METRICS (Imbalanced Test Set)")
    print(f"{'=' * 50}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  ROC AUC:    {roc_auc:.4f}")
    print(f"  PR AUC:     {pr_auc_val:.4f}")
    print(f"{'=' * 50}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": model,
        "feature_columns": available_cols,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "fraud_rate": float(y.mean()),
    }
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to: {output_path}")
    return metrics


def save_to_parquet(df: pd.DataFrame, path: str = "./data/parquet/silver_fraud_alerts"):
    """Save generated data as parquet for downstream use."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"Data saved to: {path} ({len(df):,} records)")


def main():
    parser = argparse.ArgumentParser(description="Generate imbalanced fraud transactions")
    parser.add_argument("--n-samples", type=int, default=100_000, help="Total transactions to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.0001, help="Fraud rate (default: 0.01%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-parquet", type=str, default="./data/parquet/silver_fraud_alerts")
    parser.add_argument("--output-model", type=str, default="./models/v1_imbalanced.pkl")
    args = parser.parse_args()

    print("=" * 60)
    print("  AMASTAN IMBALANCED DATA GENERATOR")
    print("=" * 60)
    print(f"\n  Samples: {args.n_samples:,}")
    print(f"  Fraud rate: {args.fraud_rate*100:.4f}% ({int(args.n_samples * args.fraud_rate)} expected fraud)")
    print(f"  Seed: {args.seed}")
    print()

    # Generate transactions
    generator = ImbalancedTransactionGenerator(
        n_samples=args.n_samples,
        fraud_rate=args.fraud_rate,
        random_seed=args.seed,
    )
    df = generator.generate_with_features()

    # Save as parquet
    save_to_parquet(df, args.output_parquet)

    # Train model
    metrics = train_on_imbalanced_data(df, args.output_model)

    print("\n" + "=" * 60)
    print("  BOOTSTRAP COMPLETE")
    print("=" * 60)
    print(f"\n  The system now has:")
    print(f"  - {len(df):,} realistic transactions")
    print(f"  - {(df['label'] == 1).sum():,} fraud cases ({(df['label'] == 1).mean()*100:.4f}%)")
    print(f"  - A trained model with F1={metrics['f1']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
    print(f"\n  These metrics are meaningful because the data reflects")
    print(f"  real-world class imbalance, not a naive 50/50 split.")
    print("=" * 60)


if __name__ == "__main__":
    main()
