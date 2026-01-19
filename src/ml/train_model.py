# src/ml/train_model.py
import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, stddev, mean, count
from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from datetime import datetime
import shutil
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

    def detect_data_drift(self):
        """Detect data drift in the incoming data"""
        try:
            # Load recent data to compare
            recent_df = self.spark.read.parquet(self.silver_path)

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
        # Check if we have enough feedback to trigger retraining
        feedback_count = self.check_feedback_availability()
        print(f"Available feedback records: {feedback_count}")

        # Also check for data drift
        drift_detected, drift_details = self.detect_data_drift()

        # Trigger retraining if we have enough feedback OR data drift is detected
        if feedback_count < 100 and not drift_detected:  # Lower threshold for testing purposes
            print(f"Not enough feedback for retraining ({feedback_count}/{100}) and no drift detected. Skipping...")
            return False

        print("Starting champion-challenger model training...")

        # Load combined training data
        dataset = self.load_and_enrich()

        # Split data for training and evaluation
        train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)

        # Train challenger model
        print("Training challenger model...")

        # Dynamically determine available feature columns from the training data
        available_cols = set(train_data.columns)
        potential_feature_cols = ["v_count", "g_dist", "avg_amount", "is_smurfing", "high_velocity_flag",
                                  "velocity_risk", "travel_risk", "high_value_risk", "d17_risk", "risk_score"]

        # Only use features that actually exist in the data
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

        # Evaluate challenger model
        challenger_metrics = self.evaluate_model(challenger_model, test_data)
        print(f"Challenger model metrics: {challenger_metrics}")

        # Load champion model if it exists and evaluate it
        champion_model_path = "models/fraud_xgb_v1"
        champion_exists = Path(champion_model_path).exists()

        if champion_exists:
            print("Evaluating champion model...")
            try:
                champion_model = Pipeline.load(champion_model_path)
                champion_metrics = self.evaluate_model(champion_model, test_data)
                print(f"Champion model metrics: {champion_metrics}")

                # Compare models and decide which to promote
                if challenger_metrics["f1_score"] > champion_metrics["f1_score"]:
                    print("✅ Challenger model performs better. Promoting to champion...")
                    # Backup old champion
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_path = f"models/fraud_xgb_v1_backup_{timestamp}"
                    if Path(champion_model_path).exists():
                        shutil.copytree(champion_model_path, backup_path)
                        print(f"Backed up old champion to {backup_path}")

                    # Save new champion
                    challenger_model.write().overwrite().save(champion_model_path)
                    print(f"✅ New champion model saved to {champion_model_path}")
                    return True
                else:
                    print("❌ Champion model performs better. Keeping current champion.")
                    return False
            except Exception as e:
                print(f"Error evaluating champion model: {e}")
                # If champion evaluation fails, save the new model anyway
                print("Saving new model as champion (champion evaluation failed)...")
                challenger_model.write().overwrite().save(champion_model_path)
                return True
        else:
            print("No champion model found. Saving first model as champion...")
            challenger_model.write().overwrite().save(champion_model_path)
            return True

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
