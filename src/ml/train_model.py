# src/ml/train_model.py
import os
import uuid
import json
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, stddev, mean, count, current_timestamp, expr, to_timestamp
from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from datetime import datetime
from pathlib import Path
from pyspark.sql.types import DoubleType, IntegerType, FloatType
from shared.utils import get_sqlite_connection


class DriftDetector:
    """Class to detect data drift in streaming fraud detection using statistical tests"""

    def __init__(self, threshold=0.05, ks_alpha=0.05):
        self.threshold = threshold  # For simple statistical differences
        self.ks_alpha = ks_alpha   # Significance level for KS test
        self.reference_samples = {}  # Store samples for KS test
        self.reference_stats = {}    # Store stats for simple comparison

    def collect_samples(self, df, sample_size=1000):
        """Collect samples for statistical testing"""
        import random
        samples = {}
        numeric_cols = [field.name for field in df.schema.fields
                       if isinstance(field.dataType, (DoubleType, IntegerType, FloatType))]

        for col_name in numeric_cols:
            # Sample data from the column
            sampled_df = df.select(col(col_name)).filter(col(col_name).isNotNull()).sample(fraction=min(sample_size/df.count(), 1.0))
            samples[col_name] = [row[0] for row in sampled_df.collect()]

        return samples

    def calculate_statistics(self, df):
        """Calculate statistical measures for a dataframe"""
        stats = {}
        numeric_cols = [field.name for field in df.schema.fields
                       if isinstance(field.dataType, (DoubleType, IntegerType, FloatType))]

        for col_name in numeric_cols:
            col_stats = df.agg(
                mean(col(col_name)).alias(f"{col_name}_mean"),
                stddev(col(col_name)).alias(f"{col_name}_stddev"),
                count(col(col_name)).alias(f"{col_name}_count")
            ).collect()[0]

            stats[col_name] = {
                "mean": col_stats[f"{col_name}_mean"],
                "stddev": col_stats[f"{col_name}_stddev"],
                "count": col_stats[f"{col_name}_count"]
            }

        return stats

    def detect_drift(self, current_stats, reference_stats=None):
        """Detect drift using both statistical comparison and KS test"""
        import scipy.stats as stats

        # Collect samples for KS test
        current_samples = self.collect_samples(self.current_df) if hasattr(self, 'current_df') else {}

        if not reference_stats:
            # Store current stats and samples as reference
            self.reference_stats = current_stats
            self.reference_samples = current_samples
            return False, "Reference stats and samples initialized"

        drift_detected = False
        drift_details = []

        # 1. Simple statistical comparison (mean/std changes)
        for col_name, curr_stat in current_stats.items():
            if col_name in reference_stats:
                ref_stat = reference_stats[col_name]

                # Calculate relative change in mean
                if ref_stat["mean"] != 0 and ref_stat["mean"] is not None:
                    mean_change = abs(curr_stat["mean"] - ref_stat["mean"]) / abs(ref_stat["mean"])
                elif curr_stat["mean"] != 0 and curr_stat["mean"] is not None:
                    mean_change = abs(curr_stat["mean"] - ref_stat["mean"])
                else:
                    mean_change = 0

                # Calculate relative change in std dev
                if ref_stat["stddev"] != 0 and ref_stat["stddev"] is not None:
                    std_change = abs(curr_stat["stddev"] - ref_stat["stddev"]) / abs(ref_stat["stddev"])
                elif curr_stat["stddev"] != 0 and curr_stat["stddev"] is not None:
                    std_change = abs(curr_stat["stddev"] - ref_stat["stddev"])
                else:
                    std_change = 0

                if mean_change > self.threshold or std_change > self.threshold:
                    drift_detected = True
                    drift_details.append({
                        "column": col_name,
                        "test_type": "statistical_comparison",
                        "mean_change": mean_change,
                        "std_change": std_change,
                        "current_mean": curr_stat["mean"],
                        "reference_mean": ref_stat["mean"],
                        "significance": "high" if (mean_change > self.threshold or std_change > self.threshold) else "low"
                    })

        # 2. Kolmogorov-Smirnov test for distribution similarity
        for col_name, current_sample in current_samples.items():
            if col_name in self.reference_samples:
                ref_sample = self.reference_samples[col_name]

                if len(current_sample) > 10 and len(ref_sample) > 10:  # Minimum sample size for KS test
                    try:
                        ks_statistic, p_value = stats.ks_2samp(ref_sample, current_sample)

                        if p_value < self.ks_alpha:  # Reject null hypothesis (distributions are different)
                            drift_detected = True
                            drift_details.append({
                                "column": col_name,
                                "test_type": "kolmogorov_smirnov",
                                "ks_statistic": ks_statistic,
                                "p_value": p_value,
                                "significant": True,
                                "message": f"Distribution drift detected (p={p_value:.4f} < alpha={self.ks_alpha})"
                            })
                        else:
                            drift_details.append({
                                "column": col_name,
                                "test_type": "kolmogorov_smirnov",
                                "ks_statistic": ks_statistic,
                                "p_value": p_value,
                                "significant": False,
                                "message": f"No significant distribution drift (p={p_value:.4f} >= alpha={self.ks_alpha})"
                            })
                    except Exception as e:
                        print(f"Error in KS test for {col_name}: {e}")
                        drift_details.append({
                            "column": col_name,
                            "test_type": "kolmogorov_smirnov",
                            "error": str(e),
                            "significant": False,
                            "message": f"KS test failed: {e}"
                        })

        return drift_detected, drift_details


class FraudModelTrainer:
    def __init__(self, silver_path="./data/parquet/silver_fraud_alerts", feedback_db_path="./data/feedback.db"):
        self.spark = SparkSession.builder.appName("TunisianFraud-ModelTrainer").getOrCreate()
        self.silver_path = silver_path
        self.feedback_db_path = feedback_db_path
        self.drift_detector = DriftDetector(threshold=0.1)

    @staticmethod
    def _parse_float_env(name, default):
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _parse_int_env(name, default):
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default


    def _ensure_model_registry(self):
        conn = get_sqlite_connection(self.feedback_db_path)
        cursor = conn.cursor()
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
        cursor.execute("PRAGMA table_info(model_registry)")
        registry_columns = {row[1] for row in cursor.fetchall()}
        if "feature_importance" not in registry_columns:
            cursor.execute("ALTER TABLE model_registry ADD COLUMN feature_importance TEXT")
        conn.commit()
        conn.close()

    def _get_current_champion(self):
        conn = get_sqlite_connection(self.feedback_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT version_id, model_path, f1_score, auc, promoted_at
            FROM model_registry
            WHERE is_champion = 1
            ORDER BY promoted_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "version_id": row[0],
            "model_path": row[1],
            "f1_score": row[2],
            "auc": row[3],
            "promoted_at": row[4]
        }

    def _record_model_registry_entry(
        self,
        version_id,
        model_path,
        f1_score,
        auc,
        is_champion,
        promoted_at,
        training_samples_count,
        feature_importance
    ):
        conn = get_sqlite_connection(self.feedback_db_path)
        cursor = conn.cursor()
        if is_champion:
            cursor.execute("UPDATE model_registry SET is_champion = 0 WHERE is_champion = 1")
        cursor.execute("""
            INSERT INTO model_registry
            (version_id, model_path, f1_score, auc, is_champion, promoted_at, training_samples_count, feature_importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version_id,
            model_path,
            f1_score,
            auc,
            1 if is_champion else 0,
            promoted_at,
            training_samples_count,
            feature_importance
        ))
        conn.commit()
        conn.close()

    def check_feedback_availability(self):
        """Check if enough human feedback is available to retrain"""
        try:
            conn = get_sqlite_connection(self.feedback_db_path)
            cursor = conn.cursor()

            # Count total feedback records
            cursor.execute("SELECT COUNT(*) FROM feedback_labels WHERE analyst_label IS NOT NULL")
            total_feedback = cursor.fetchone()[0]

            conn.close()
            return total_feedback
        except Exception as e:
            print(f"Error checking feedback availability: {e}")
            return 0

    def load_feedback_data(self):
        """Load human-verified feedback data for retraining"""
        try:
            conn = get_sqlite_connection(self.feedback_db_path)
            cursor = conn.cursor()

            # Get feedback data with proper label mapping
            cursor.execute("""
                SELECT
                    transaction_id,
                    user_id,
                    amount_tnd,
                    governorate,
                    payment_method,
                    ml_probability,
                    CASE
                        WHEN analyst_label = 'Confirmed Fraud' THEN 1
                        WHEN analyst_label = 'False Positive' THEN 0
                        ELSE -1  -- Invalid/missing label
                    END as verified_label
                FROM feedback_labels
                WHERE analyst_label IS NOT NULL
                AND (analyst_label = 'Confirmed Fraud' OR analyst_label = 'False Positive')
            """)

            feedback_records = cursor.fetchall()
            conn.close()

            # Convert to DataFrame if records exist
            if feedback_records:
                # Create a temporary view for SQL operations
                feedback_df = self.spark.createDataFrame(
                    feedback_records,
                    ["transaction_id", "user_id", "amount_tnd", "governorate", "payment_method", "ml_probability", "verified_label"]
                )

                # Process feedback data to create features
                enriched_feedback = feedback_df.withColumn("is_smurfing", when(col("amount_tnd").between(1400, 1500), 1).otherwise(0)) \
                                              .withColumn("high_velocity_flag", lit(0))  # Placeholder - would need to compute from history
                return enriched_feedback
            else:
                return None
        except Exception as e:
            print(f"Error loading feedback data: {e}")
            return None

    def load_and_enrich(self):
        """Load and enrich data - PRIORITY: Use only verified feedback data for training"""
        # Load human feedback data (this is the only data with verified labels)
        feedback_df = self.load_feedback_data()

        if feedback_df is not None and feedback_df.count() > 0:
            print(f"Loaded {feedback_df.count()} records from human feedback (verified labels)")

            # Use only feedback data with verified labels for training
            feedback_cols = feedback_df.columns
            if "verified_label" in feedback_cols:
                # Rename verified_label to label for consistency
                combined_df = feedback_df.withColumnRenamed("verified_label", "label")
            else:
                combined_df = feedback_df

            print(f"Using {combined_df.count()} verified records for model training")
        else:
            print("No verified feedback data available for training")

            # As a fallback (for initial training), we can use silver data with heuristic labels
            # But this should be avoided in production once feedback is available
            try:
                silver_df = self.spark.read.parquet(self.silver_path)
                print(f"Using {silver_df.count()} records from silver layer with heuristic labels (NOT RECOMMENDED in production)")

                # Create heuristic labels based on ml_probability threshold
                # This is only for initial model training before feedback is available
                combined_df = silver_df.withColumn(
                    "label",
                    when(col("ml_probability") > 0.9, 1).otherwise(0)  # Higher threshold for heuristic labeling
                )

                # Only use records with heuristic labels
                combined_df = combined_df.filter(col("label").isin([0, 1]))

            except Exception as e:
                print(f"Error: Could not load silver layer data as fallback: {e}")
                raise Exception("No data available for training - need either feedback data or silver layer data")

        return combined_df

    def evaluate_model(self, model, test_data):
        """Evaluate model performance"""
        predictions = model.transform(test_data)

        # Calculate AUC
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

        # Calculate precision, recall, and F1 manually
        from pyspark.sql.functions import sum as spark_sum, count

        # Calculate TP, FP, TN, FN
        tp = predictions.filter((col("prediction") == 1) & (col("label") == 1)).count()
        fp = predictions.filter((col("prediction") == 1) & (col("label") == 0)).count()
        tn = predictions.filter((col("prediction") == 0) & (col("label") == 0)).count()
        fn = predictions.filter((col("prediction") == 0) & (col("label") == 1)).count()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }

    @staticmethod
    def get_feature_scores(model):
        """Extract feature importance scores from an XGBoost pipeline model."""
        try:
            feature_cols = None
            xgb_model = None

            if hasattr(model, "stages"):
                for stage in model.stages:
                    if hasattr(stage, "getInputCols"):
                        feature_cols = stage.getInputCols()
                    if hasattr(stage, "get_booster"):
                        xgb_model = stage
            else:
                xgb_model = model

            if not xgb_model:
                return []

            booster = xgb_model.get_booster()
            scores = booster.get_score(importance_type="gain")
            if not scores:
                return []

            items = []
            for key, score in scores.items():
                if key.startswith("f") and key[1:].isdigit() and feature_cols:
                    idx = int(key[1:])
                    if idx < len(feature_cols):
                        items.append((feature_cols[idx], float(score)))
                    else:
                        items.append((key, float(score)))
                else:
                    items.append((key, float(score)))

            items.sort(key=lambda item: item[1], reverse=True)
            return items
        except Exception as e:
            print(f"Error extracting feature scores: {e}")
            return []

    def detect_data_drift(self):
        """Detect data drift in the incoming data"""
        try:
            # Load recent data to compare
            recent_df = self.spark.read.parquet(self.silver_path)

            if "timestamp" in recent_df.columns:
                window_hours = max(1, self._parse_int_env("DRIFT_WINDOW_HOURS", 24))
                recent_df = recent_df.withColumn("event_time", to_timestamp(col("timestamp")))
                recent_df = recent_df.filter(
                    col("event_time") >= (current_timestamp() - expr(f"INTERVAL {window_hours} HOURS"))
                )

            # Calculate statistics for recent data
            recent_stats = self.drift_detector.calculate_statistics(recent_df)

            # Store the dataframe for sample collection in drift detector
            self.drift_detector.current_df = recent_df

            # Detect drift
            drift_detected, drift_details = self.drift_detector.detect_drift(
                recent_stats, self.drift_detector.reference_stats
            )

            if drift_detected:
                print(f"⚠️ Data drift detected: {drift_details}")
                return True, drift_details
            else:
                print("✅ No significant data drift detected")
                return False, []

        except Exception as e:
            print(f"Error detecting data drift: {e}")
            return False, []

    def train_champion_challenger(self):
        """Implement champion-challenger model retraining logic"""
        self._ensure_model_registry()

        feedback_count = self.check_feedback_availability()
        print(f"Available feedback records: {feedback_count}")

        drift_detected, drift_details = self.detect_data_drift()

        min_feedback = max(1, int(self._parse_float_env("RETRAIN_MIN_FEEDBACK", 100)))
        if feedback_count < min_feedback and not drift_detected:
            print(
                "Not enough feedback for retraining "
                f"({feedback_count}/{min_feedback}) and no drift detected. Skipping..."
            )
            return False

        print("Starting champion-challenger model training...")

        dataset = self.load_and_enrich()
        train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)
        training_samples_count = train_data.count()

        print("Training challenger model...")

        available_cols = set(train_data.columns)
        potential_feature_cols = [
            "v_count",
            "g_dist",
            "avg_amount",
            "is_smurfing",
            "high_velocity_flag",
            "velocity_risk",
            "travel_risk",
            "high_value_risk",
            "d17_risk",
            "risk_score"
        ]

        feature_cols = [col for col in potential_feature_cols if col in available_cols and col != "label"]

        print(f"Using features: {feature_cols}")

        if not feature_cols:
            raise ValueError("No valid feature columns found in training data")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

        xgb = SparkXGBClassifier(
            featuresCol="features",
            labelCol="label",
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1
        )

        pipeline = Pipeline(stages=[assembler, xgb])
        challenger_model = pipeline.fit(train_data)

        challenger_metrics = self.evaluate_model(challenger_model, test_data)
        feature_scores = self.get_feature_scores(challenger_model)
        feature_importance = json.dumps([
            {"feature": name, "score": float(score)}
            for name, score in feature_scores
        ])
        print(f"Challenger model metrics: {challenger_metrics}")

        champion_entry = self._get_current_champion()
        champion_metrics = None
        if champion_entry:
            print("Evaluating champion model...")
            try:
                champion_model = PipelineModel.load(champion_entry["model_path"])
                champion_metrics = self.evaluate_model(champion_model, test_data)
                print(f"Champion model metrics: {champion_metrics}")
            except Exception as e:
                print(f"Error evaluating champion model: {e}")

        promotion_threshold = self._parse_float_env("CHAMPION_PROMOTION_THRESHOLD", 0.02)
        promote = False
        if not champion_entry or not champion_metrics:
            promote = True
        else:
            improvement = challenger_metrics["f1_score"] - champion_metrics["f1_score"]
            promote = improvement >= promotion_threshold

        version_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        model_path = str(Path("models") / "registry" / f"fraud_xgb_{version_id}")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        challenger_model.write().overwrite().save(model_path)

        promoted_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") if promote else None
        self._record_model_registry_entry(
            version_id=version_id,
            model_path=model_path,
            f1_score=challenger_metrics["f1_score"],
            auc=challenger_metrics["auc"],
            is_champion=promote,
            promoted_at=promoted_at,
            training_samples_count=training_samples_count,
            feature_importance=feature_importance
        )

        if promote:
            print("Challenger model promoted to champion.")
            return True

        print("Champion model retained. Challenger registered for audit.")
        return False

    def schedule_retraining(self, interval_minutes=60):
        """Schedule periodic retraining of the model"""
        import time
        import threading

        def retrain_loop():
            while True:
                try:
                    print(f"Checking for retraining opportunity at {datetime.now()}")
                    success = self.train_champion_challenger()
                    if success:
                        print(f"✅ Model retrained successfully at {datetime.now()}")
                    else:
                        print(f"ℹ️  No retraining needed at {datetime.now()}")

                    # Wait for the specified interval before next check
                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    print(f"Error in retraining loop: {e}")
                    time.sleep(interval_minutes * 60)  # Wait before retrying even if there's an error

        # Start the retraining thread
        retrain_thread = threading.Thread(target=retrain_loop, daemon=True)
        retrain_thread.start()
        return retrain_thread


if __name__ == "__main__":
    trainer = FraudModelTrainer()

    # Option 1: Run a single training cycle
    success = trainer.train_champion_challenger()
    if success:
        print("✅ Champion-challenger training cycle completed successfully")
    else:
        print("ℹ️  No model update performed")

    # Option 2: Start continuous retraining (uncomment to use)
    # print("Starting continuous retraining scheduler...")
    # trainer.schedule_retraining(interval_minutes=30)  # Check every 30 minutes

    # Keep the main thread alive if running continuous retraining
    # import time
    # while True:
    #     time.sleep(60)  # Sleep for 1 minute
