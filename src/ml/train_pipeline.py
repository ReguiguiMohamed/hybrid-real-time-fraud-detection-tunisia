"""
Scikit-Learn Pipeline for Fraud Detection Model Training
Ensures NO data leakage: all preprocessing parameters are fit ONLY on training data.

Key differences from the notebook EDA approach:
- EDA is done on the ENTIRE dataset (data leakage risk)
- Training uses a proper Pipeline with train/test split BEFORE any scaling
- Feature scaling (StandardScaler) is fit on train data, then applied to test
- SMOTE is applied ONLY to training data (never to test data)

Usage:
    python src/ml/train_pipeline.py
"""
import os
import sys
import json
import logging
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from compliance.change_audit import append_change_audit_event
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc as pr_auc,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.ensemble import GradientBoostingClassifier
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Proper ML pipeline for fraud detection with NO data leakage.

    Architecture:
    1. Load data (feedback + silver parquet)
    2. Split train/test FIRST (before any preprocessing)
    3. Build preprocessing pipeline (fit on train ONLY)
    4. Apply SMOTE to train ONLY
    5. Train model on preprocessed train
    6. Evaluate on untouched test set
    7. Save pipeline + metrics
    """

    def __init__(
        self,
        feedback_db_path: str = "./data/feedback.db",
        silver_parquet_path: str = "./data/parquet/silver_fraud_alerts",
        model_output_path: str = "./models/registry",
    ):
        self.feedback_db_path = feedback_db_path
        self.silver_parquet_path = silver_parquet_path
        self.model_output_path = Path(model_output_path)
        self.model_output_path.mkdir(parents=True, exist_ok=True)

        self.pipeline = None
        self.metrics = {}

    # ==================== Data Loading ====================

    def load_feedback_data(self) -> pd.DataFrame:
        """Load human-verified feedback data from SQLite."""
        if not Path(self.feedback_db_path).exists():
            logger.warning("Feedback database not found")
            return pd.DataFrame()

        conn = sqlite3.connect(self.feedback_db_path)
        try:
            query = """
                SELECT
                    hra.transaction_id,
                    hra.amount_tnd,
                    hra.governorate,
                    hra.payment_method,
                    hra.ml_probability,
                    hra.risk_score,
                    hra.v_count,
                    hra.g_dist,
                    CASE
                        WHEN fl.analyst_label = 'Confirmed Fraud' THEN 1
                        WHEN fl.analyst_label = 'False Positive' THEN 0
                    END as label
                FROM high_risk_alerts hra
                INNER JOIN feedback_labels fl
                    ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IN ('Confirmed Fraud', 'False Positive')
            """
            df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    def load_silver_data(self) -> pd.DataFrame:
        """Load silver layer parquet data with heuristic labels (fallback only)."""
        if not Path(self.silver_parquet_path).exists():
            logger.warning("Silver parquet data not found")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(self.silver_parquet_path)

            # Create heuristic labels from ML probability (NOT recommended for production)
            if "ml_probability" in df.columns:
                df["label"] = (df["ml_probability"] > 0.9).astype(int)
            else:
                logger.warning("No ml_probability column found, cannot create labels")
                return pd.DataFrame()

            return df
        except Exception as e:
            logger.error(f"Error loading silver data: {e}")
            return pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """Load and merge data sources. Priority: feedback > silver."""
        feedback_df = self.load_feedback_data()
        silver_df = self.load_silver_data()

        if not feedback_df.empty:
            logger.info(f"Loaded {len(feedback_df)} human-verified feedback records")
            if not silver_df.empty:
                # Combine feedback with a sample of silver data
                # Only use silver data with high-confidence labels
                silver_confident = silver_df[silver_df["ml_probability"] > 0.95]
                combined = pd.concat([feedback_df, silver_confident], ignore_index=True)
                logger.info(f"Combined dataset: {len(combined)} records")
                return combined
            return feedback_df
        elif not silver_df.empty:
            logger.warning(f"Using {len(silver_df)} silver records with heuristic labels (NOT RECOMMENDED)")
            return silver_df
        else:
            raise ValueError("No data available for training")

    # ==================== Pipeline Construction ====================

    def build_preprocessing_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Build preprocessing pipeline.
        IMPORTANT: This will be FIT ONLY on training data.
        """
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("label", "transaction_id")]

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c != "transaction_id"]

        logger.info(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
        logger.info(f"Numeric: {numeric_cols}")
        logger.info(f"Categorical: {categorical_cols}")

        # Numeric pipeline: impute + scale
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        # Categorical pipeline: impute + one-hot encode
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        return preprocessor

    def build_full_pipeline(self, preprocessor):
        """
        Build the full pipeline: preprocessing + SMOTE + model.
        SMOTE is applied ONLY during fit on training data.
        """
        # Gradient Boosting (handles imbalanced data better than XGBoost for smaller datasets)
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )

        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as exc:
            raise ImportError(
                "Training with SMOTE requires compatible imbalanced-learn and scikit-learn versions. "
                "Install the pinned project requirements before running train_pipeline.py."
            ) from exc

        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),  # Applied ONLY to train
            ("classifier", model),
        ])

        return pipeline

    # ==================== Training & Evaluation ====================

    def train_and_evaluate(self, df: pd.DataFrame) -> dict:
        """
        Full training and evaluation workflow.

        Key: Split BEFORE any preprocessing to prevent data leakage.
        """
        # Drop transaction_id (not a feature)
        feature_cols = [c for c in df.columns if c not in ("transaction_id", "label")]
        X = df[feature_cols]
        y = df["label"]

        # Check class distribution
        fraud_count = y.sum()
        total_count = len(y)
        fraud_rate = fraud_count / total_count if total_count > 0 else 0
        logger.info(f"Class distribution: {fraud_count}/{total_count} fraud ({fraud_rate*100:.2f}%)")

        # Split FIRST (before any preprocessing)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"Train fraud rate: {y_train.mean()*100:.2f}%")
        logger.info(f"Test fraud rate: {y_test.mean()*100:.2f}%")

        # Build preprocessing pipeline
        preprocessor = self.build_preprocessing_pipeline(X_train)

        # Build full pipeline (SMOTE is only applied during .fit())
        self.pipeline = self.build_full_pipeline(preprocessor)

        # Fit pipeline (SMOTE applied to train only, scaler fit on train only)
        logger.info("Fitting pipeline on training data...")
        self.pipeline.fit(X_train, y_train)

        # Evaluate on untouched test set
        logger.info("Evaluating on test set...")
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        # Comprehensive metrics
        self.metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": pr_auc(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_columns": feature_cols,
        }

        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall:    {self.metrics['recall']:.4f}")
        logger.info(f"F1:        {self.metrics['f1']:.4f}")
        logger.info(f"ROC AUC:   {self.metrics['roc_auc']:.4f}")
        logger.info(f"PR AUC:    {self.metrics['pr_auc']:.4f}")
        logger.info(f"\nConfusion Matrix:\n{self.metrics['confusion_matrix']}")

        # Precision-Recall curve data (for dashboard plotting)
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
        self.metrics["pr_curve"] = {
            "precision": precision_vals.tolist(),
            "recall": recall_vals.tolist(),
            "thresholds": thresholds.tolist(),
        }

        return self.metrics

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        """Cross-validation with stratified folds for more robust metrics."""
        feature_cols = [c for c in df.columns if c not in ("transaction_id", "label")]
        X = df[feature_cols]
        y = df["label"]

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        f1_scores = []
        pr_auc_scores = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Build and fit pipeline
            preprocessor = self.build_preprocessing_pipeline(X_train)
            pipeline = self.build_full_pipeline(preprocessor)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred, zero_division=0)
            pr = pr_auc(y_test, y_prob)

            f1_scores.append(f1)
            pr_auc_scores.append(pr)

            logger.info(f"Fold {fold+1}: F1={f1:.4f}, PR-AUC={pr:.4f}")

        cv_metrics = {
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "pr_auc_mean": np.mean(pr_auc_scores),
            "pr_auc_std": np.std(pr_auc_scores),
            "f1_per_fold": f1_scores,
            "pr_auc_per_fold": pr_auc_scores,
        }

        logger.info(f"\nCross-Validation ({n_splits} folds):")
        logger.info(f"  F1:    {cv_metrics['f1_mean']:.4f} (+/- {cv_metrics['f1_std']:.4f})")
        logger.info(f"  PR-AUC: {cv_metrics['pr_auc_mean']:.4f} (+/- {cv_metrics['pr_auc_std']:.4f})")

        return cv_metrics

    # ==================== Saving & Registry ====================

    def save_model(self, version_id: str = None) -> str:
        """Save the trained pipeline and register in model registry."""
        if self.pipeline is None:
            raise RuntimeError("No trained pipeline to save")

        if version_id is None:
            version_id = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")

        model_dir = self.model_output_path / f"fraud_pipeline_{version_id}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save pipeline
        pipeline_path = model_dir / "pipeline.pkl"
        joblib.dump(self.pipeline, pipeline_path)
        logger.info(f"Pipeline saved to {pipeline_path}")

        # Save metrics
        metrics_path = model_dir / "metrics.json"
        # Remove non-serializable items
        serializable_metrics = {k: v for k, v in self.metrics.items() if k not in ("pr_curve",)}
        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        # Save feature importance
        if hasattr(self.pipeline, "named_steps") and hasattr(self.pipeline.named_steps["classifier"], "feature_importances_"):
            importances = self.pipeline.named_steps["classifier"].feature_importances_
            feature_names = self.metrics.get("feature_columns", [])
            feature_importance = [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(feature_names, importances)
            ]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

            with open(model_dir / "feature_importance.json", "w") as f:
                json.dump(feature_importance, f, indent=2)

        # Register in model registry
        self._register_in_registry(version_id, str(model_dir))

        return str(model_dir)

    def _register_in_registry(self, version_id: str, model_path: str):
        """Register the model in the SQLite model registry."""
        from src.shared.utils import get_sqlite_connection

        approved_by = os.getenv("MODEL_PROMOTION_APPROVED_BY")
        if not approved_by or approved_by.strip().lower() in {"system", "automation", "auto", "scheduler"}:
            append_change_audit_event({
                "event_type": "MODEL_PROMOTION_BLOCKED",
                "actor": os.getenv("MODEL_PROMOTION_USER", "system"),
                "approved_by": None,
                "entity_type": "MODEL",
                "entity_id": version_id,
                "action": "BLOCK_PROMOTION",
                "previous_state": None,
                "new_state": {
                    "version_id": version_id,
                    "model_path": model_path,
                    "f1_score": self.metrics.get("f1", 0.0),
                    "auc": self.metrics.get("pr_auc", 0.0),
                    "training_samples_count": self.metrics.get("train_samples", 0),
                },
                "promotion_trigger": "train_pipeline_register",
                "performance_delta": None,
                "justification": "Standalone model registration attempted without a human approver.",
            })
            raise RuntimeError("MODEL_PROMOTION_APPROVED_BY must identify a human approver before champion registration.")

        conn = get_sqlite_connection(self.feedback_db_path)
        cursor = conn.cursor()

        # Create registry table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                version_id TEXT PRIMARY KEY,
                model_path TEXT NOT NULL,
                f1_score REAL,
                auc REAL,
                is_champion INTEGER DEFAULT 0,
                promoted_at DATETIME,
                training_samples_count INTEGER,
                feature_importance TEXT
            )
        """)

        cursor.execute("""
            SELECT version_id, model_path, f1_score, auc, promoted_at
            FROM model_registry
            WHERE is_champion = 1
            ORDER BY promoted_at DESC
            LIMIT 1
        """)
        previous_champion = cursor.fetchone()
        previous_state = None
        if previous_champion:
            previous_state = {
                "version_id": previous_champion[0],
                "model_path": previous_champion[1],
                "f1_score": previous_champion[2],
                "auc": previous_champion[3],
                "promoted_at": previous_champion[4],
            }

        # Demote current champion
        cursor.execute("UPDATE model_registry SET is_champion = 0 WHERE is_champion = 1")

        # Insert new model as champion
        f1 = self.metrics.get("f1", 0.0)
        roc_auc = self.metrics.get("roc_auc", 0.0)
        pr_auc_val = self.metrics.get("pr_auc", 0.0)
        train_samples = self.metrics.get("train_samples", 0)

        cursor.execute("""
            INSERT INTO model_registry
            (version_id, model_path, f1_score, auc, is_champion, promoted_at, training_samples_count, feature_importance)
            VALUES (?, ?, ?, ?, 1, ?, ?, ?)
        """, (
            version_id,
            model_path,
            f1,
            pr_auc_val,  # Use PR-AUC for AUC column (more meaningful for imbalanced data)
            datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
            train_samples,
            json.dumps(self.metrics.get("feature_columns", [])),
        ))

        conn.commit()
        conn.close()
        append_change_audit_event({
            "event_type": "MODEL_PROMOTION",
            "actor": approved_by.strip(),
            "approved_by": approved_by.strip(),
            "entity_type": "MODEL",
            "entity_id": version_id,
            "action": "PROMOTE",
            "previous_state": previous_state,
            "new_state": {
                "version_id": version_id,
                "model_path": model_path,
                "f1_score": f1,
                "auc": pr_auc_val,
                "training_samples_count": train_samples,
            },
            "promotion_trigger": "train_pipeline_register",
            "performance_delta": None,
            "justification": "Human-approved champion registration from offline training pipeline.",
        })
        logger.info(f"Model registered as champion: {version_id} (F1={f1:.4f}, PR-AUC={pr_auc_val:.4f})")


def main():
    """Main training entry point."""
    logger.info("=" * 60)
    logger.info("  Amastan Fraud Detection - Pipeline Training")
    logger.info("=" * 60)

    pipeline_trainer = FraudDetectionPipeline()

    # Load data
    df = pipeline_trainer.load_data()
    if df.empty:
        logger.error("No data available for training")
        sys.exit(1)

    # Train and evaluate
    metrics = pipeline_trainer.train_and_evaluate(df)

    # Cross-validation for robust metrics
    cv_metrics = pipeline_trainer.cross_validate(df)

    # Save model
    version_id = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    model_path = pipeline_trainer.save_model(version_id)

    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  PR-AUC:   {metrics['pr_auc']:.4f}")
    logger.info(f"  CV F1:    {cv_metrics['f1_mean']:.4f} (+/- {cv_metrics['f1_std']:.4f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
