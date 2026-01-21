# src/streaming/consumer.py
import os
import logging
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set HADOOP_HOME from environment variable
hadoop_home = os.getenv('HADOOP_HOME', r'C:\hadoop-3.4.2')
os.environ['HADOOP_HOME'] = hadoop_home

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, count, approx_count_distinct, when, lit, to_timestamp, expr
from pyspark.sql.types import DoubleType
from shared.schemas import Transaction, TRANSACTION_SPARK_SCHEMA
from shared.risk_config import RISK_WEIGHTS, CBDC_PILOT_GOVERNORATES, D17_SOFT_LIMIT, D17_VELOCITY_CAP
from shared.quality_gates import validate_transaction_quality, apply_d17_rule
from shared.utils import make_authenticated_request, log_failed_alert, retry_failed_alerts, get_sqlite_connection

# Use the schema from the shared module to ensure consistency
schema = TRANSACTION_SPARK_SCHEMA

class FraudProcessor:
    def __init__(self, kafka_bootstrap=None):
        # Initializing with Kafka support (Delta Lake config removed to avoid streaming conflicts)
        self.spark = SparkSession.builder \
            .appName("Tunisia-Fraud-Silver-Layer") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1") \
            .config("spark.sql.streaming.checkpointLocation", "./tmp/checkpoint") \
            .getOrCreate()
        if kafka_bootstrap is None:
            kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
        self.kafka_bootstrap = kafka_bootstrap

        # Load XGBoost model for real-time inference
        try:
            from xgboost.spark import SparkXGBClassifierModel
            self.ml_model = SparkXGBClassifierModel.load("models/fraud_xgb_v1")
            print("✅ XGBoost Model loaded for Real-Time Inference")
        except Exception as e:
            print(f"⚠️ Fallback to Rule-Based Scoring. Model not available: {e}")
            self.ml_model = None
        self._batch_counter = 0
        self._feedback_db_path = "./data/feedback.db"

    def start_dlq_retry_worker(self):
        if getattr(self, "_dlq_retry_thread", None) and self._dlq_retry_thread.is_alive():
            return

        interval_env = os.getenv("DLQ_RETRY_INTERVAL_SECONDS", "60")
        max_attempts_env = os.getenv("DLQ_RETRY_MAX_ATTEMPTS", "3")
        try:
            interval = max(1, int(interval_env))
        except ValueError:
            interval = 60
        try:
            max_attempts = max(1, int(max_attempts_env))
        except ValueError:
            max_attempts = 3
        self._dlq_retry_stop = threading.Event()

        def retry_loop():
            while not self._dlq_retry_stop.is_set():
                try:
                    retry_failed_alerts(max_attempts=max_attempts)
                except Exception:
                    logging.exception("DLQ retry worker encountered an error")
                self._dlq_retry_stop.wait(interval)

        self._dlq_retry_thread = threading.Thread(
            target=retry_loop,
            name="dlq-retry-worker",
            daemon=True
        )
        self._dlq_retry_thread.start()

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

    def _load_sampling_config(self):
        random_sample_rate = self._parse_float_env("RANDOM_SAMPLE_RATE", 0.01)
        random_sample_max_prob = self._parse_float_env("RANDOM_SAMPLE_MAX_PROB", 0.1)
        random_sample_rate = max(0.0, min(random_sample_rate, 1.0))
        random_sample_max_prob = max(0.0, min(random_sample_max_prob, 1.0))
        return random_sample_rate, random_sample_max_prob

    def _load_alerting_config(self):
        max_workers = max(1, self._parse_int_env("THREAD_POOL_SIZE", 5))
        async_timeout = max(1, self._parse_int_env("ALERT_ASYNC_TIMEOUT_SECONDS", 15))
        return max_workers, async_timeout

    def _send_alert_async(self, row, sar_gen, alert_type="high_risk", generate_sar=True):
        """Send an alert to the command center API."""
        try:
            row_dict = row.asDict()
            ml_probability = row_dict.get("ml_probability", 0.0)
            if ml_probability is None:
                ml_probability = 0.0
            ml_probability = float(ml_probability)

            report = None
            if generate_sar and sar_gen is not None:
                report = sar_gen.generate_report(row_dict, ml_probability)
                report_path = sar_gen.save_report(row_dict, report, ml_probability)
                print(f"SAR generated and saved to: {report_path}")

            alert_payload = {
                "transaction_id": str(row_dict.get("transaction_id", "unknown")),
                "user_id": str(row_dict.get("user_id", "unknown")),
                "amount_tnd": float(row_dict.get("amount_tnd", 0.0) or 0.0),
                "governorate": str(row_dict.get("governorate", "unknown")),
                "payment_method": str(row_dict.get("payment_method", "unknown")),
                "timestamp": str(row_dict.get("timestamp", "")),
                "ml_probability": ml_probability,
                "sar_report": report,
                "alert_type": alert_type
            }

            try:
                api_response = make_authenticated_request(
                    "POST",
                    "/alerts/add/",
                    payload=alert_payload,
                    timeout=5  # 5 second timeout to avoid blocking
                )

                if api_response and api_response.status_code == 200:
                    print(
                        "Alert sent to command center for transaction: "
                        f"{row_dict.get('transaction_id')} ({alert_type})"
                    )
                else:
                    if api_response:
                        error_msg = f"{api_response.status_code} - {api_response.text}"
                        error_code = str(api_response.status_code)
                    else:
                        error_msg = "No response object returned"
                        error_code = "NO_RESPONSE"

                    print(f"Failed to send alert to command center: {error_msg}")
                    log_failed_alert(row_dict, alert_payload, error_code, error_msg)
            except Exception as api_error:
                print(f"API connection error when sending alert: {api_error}")
                log_failed_alert(row_dict, alert_payload, "CONNECTION_ERROR", str(api_error))

        except Exception as e:
            try:
                row_dict = row.asDict()
            except Exception:
                row_dict = {"transaction_id": "unknown"}
            print(f"Error processing transaction {row_dict.get('transaction_id', 'unknown')}: {e}")
            log_failed_alert(row_dict, {}, "PROCESSING_ERROR", str(e))

    def _process_batch(self, batch_df, epoch_id):
        random_sample_rate, random_sample_max_prob = self._load_sampling_config()

        high_risk_rows = batch_df.filter(col("ml_probability") > 0.85).collect()
        sampled_low_risk_rows = []

        if random_sample_rate > 0:
            low_risk_df = batch_df.filter(col("ml_probability") < random_sample_max_prob)
            if random_sample_rate < 1:
                low_risk_df = low_risk_df.sample(withReplacement=False, fraction=random_sample_rate)
            sampled_low_risk_rows = low_risk_df.collect()

        if not high_risk_rows and not sampled_low_risk_rows:
            return

        from rag_engine.sar_generator import SARGenerator
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

        sar_gen = SARGenerator() if high_risk_rows else None

        max_workers, async_timeout = self._load_alerting_config()
        print(
            f"Processing {len(high_risk_rows)} high-risk alerts and "
            f"{len(sampled_low_risk_rows)} random samples with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {}

            for row in high_risk_rows:
                future = executor.submit(self._send_alert_async, row, sar_gen, "high_risk", True)
                future_to_row[future] = row

            for row in sampled_low_risk_rows:
                future = executor.submit(self._send_alert_async, row, sar_gen, "random_sample", False)
                future_to_row[future] = row

            try:
                for future in as_completed(future_to_row, timeout=async_timeout):
                    try:
                        future.result()
                    except Exception as e:
                        row = future_to_row[future]
                        try:
                            transaction_id = row.asDict().get("transaction_id", "unknown")
                        except Exception:
                            transaction_id = "unknown"
                        print(f"Error in async alert processing for transaction {transaction_id}: {e}")
            except TimeoutError:
                print("Timed out waiting for alert processing tasks to finish")

    def _check_and_trigger_retraining(self, batch_df, epoch_id):
        self._process_batch(batch_df, epoch_id)

        self._batch_counter += 1
        if self._batch_counter % 10 != 0:
            return

        try:
            conn = get_sqlite_connection(self._feedback_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback_labels WHERE analyst_label IS NOT NULL")
            feedback_count = cursor.fetchone()[0]
            conn.close()

            if feedback_count >= 50:
                print(f"Triggering model retraining based on {feedback_count} feedback records")

                retrain_response = make_authenticated_request(
                    "POST",
                    "/retrain-model/",
                    timeout=10  # 10 second timeout for retraining trigger
                )

                if retrain_response and retrain_response.status_code == 200:
                    print("ƒo. Model retraining triggered successfully")
                else:
                    if retrain_response:
                        print(
                            "Failed to trigger model retraining: "
                            f"{retrain_response.status_code} - {retrain_response.text}"
                        )
                    else:
                        print("Failed to trigger model retraining: No response received")

        except Exception as e:
            print(f"Error checking feedback for retraining: {e}")

    def process_stream(self):
        # 1. Ingest (Bronze Layer)
        raw_stream = self.spark.readStream.format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap) \
            .option("subscribe", "tunisian_transactions") \
            .load()

        # Deserialize JSON value
        json_df = raw_stream.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")

        # Apply data quality gates
        validated_df = validate_transaction_quality(json_df)

        # 2. Enrich & Score (Silver Layer)
        enriched = validated_df.withColumn("event_time", to_timestamp(col("timestamp"))) \
                         .withWatermark("event_time", "10 minutes")

        # Apply D17 rule for risk boosting
        enriched_with_d17 = apply_d17_rule(enriched)

        # Complex Windowing: Velocity + Multi-Gov
        analytics = enriched_with_d17.groupBy(
            window(col("event_time"), "5 minutes", "1 minute"),
            col("user_id")
        ).agg(
            count("transaction_id").alias("v_count"),
            approx_count_distinct("governorate").alias("g_dist"),
            lit(None).cast(DoubleType()).alias("amount_tnd")  # Placeholder for avg amount
        )

        # Calculate average amount per user in window, including payment method info for D17 rule
        analytics_with_amount = enriched_with_d17.groupBy(
            window(col("event_time"), "5 minutes", "1 minute"),
            col("user_id")
        ).agg(
            count("transaction_id").alias("v_count"),
            approx_count_distinct("governorate").alias("g_dist"),
            expr("avg(amount_tnd)").alias("avg_amount"),
            # Check if any transaction in the window used Flouci method
            expr("sum(case when payment_method = 'Flouci' then 1 else 0 end)").alias("flouci_count")
        )

        # 3. Weighted Risk Scoring (The Industrial Logic)
        scored = analytics_with_amount.withColumn(
            "velocity_risk",
            when(col("v_count") > 3, lit(1.0)).otherwise(lit(0.0))
        ).withColumn(
            "travel_risk",
            when(col("g_dist") > 1, lit(1.0)).otherwise(lit(0.0))
        ).withColumn(
            "high_value_risk",
            when(col("avg_amount") > 5000, lit(1.0)).otherwise(lit(0.0))  # Threshold for high value
        ).withColumn(
            "d17_risk",
            when((col("avg_amount") > 2000) & (col("flouci_count") > 0), lit(1.0)).otherwise(lit(0.0))
        ).withColumn(
            "risk_score",
            (col("velocity_risk") * RISK_WEIGHTS["velocity"]) +
            (col("travel_risk") * RISK_WEIGHTS["travel"]) +
            (col("high_value_risk") * RISK_WEIGHTS["high_value"]) +
            (col("d17_risk") * RISK_WEIGHTS["d17_limit"])
        )

        # Prepare features for ML model regardless of model availability to ensure consistent schema
        features_df = scored.withColumn("is_smurfing", when(col("avg_amount").between(1400, 1500), 1).otherwise(0)) \
                            .withColumn("high_velocity_flag", when(col("v_count") > D17_VELOCITY_CAP, 1).otherwise(0))

        # Apply ML inference if model is available
        if self.ml_model:
            from pyspark.ml.feature import VectorAssembler

            # Create feature vector for ML model
            assembler = VectorAssembler(
                inputCols=["v_count", "g_dist", "avg_amount", "is_smurfing", "high_velocity_flag"],
                outputCol="features"
            )

            # Transform the streaming data on-the-fly
            assembled_df = assembler.transform(features_df)
            predictions = self.ml_model.transform(assembled_df)
            final_df = predictions.withColumnRenamed("prediction", "ml_prediction") \
                                 .withColumnRenamed("probability", "ml_probability")
        else:
            # Fallback to rule-based scoring but maintain consistent schema
            final_df = features_df.withColumn("ml_prediction", lit(-1)) \
                                 .withColumn("ml_probability", lit(0.0))

        # For performance, use foreachBatch to handle SAR generation and alerting asynchronously.

        # 4. Persistence: Using Parquet for streaming (Delta Lake for batch operations)
        # Due to compatibility issues between Spark 4.1.1 and Delta Lake 4.0.1 for streaming sinks
        query = final_df.writeStream \
            .format("parquet") \
            .outputMode("append") \
            .option("path", "./data/parquet/silver_fraud_alerts") \
            .option("checkpointLocation", "./tmp/checkpoint/silver_fraud") \
            .foreachBatch(self._check_and_trigger_retraining) \
            .start()

        self.start_dlq_retry_worker()

        return query

if __name__ == "__main__":
    processor = FraudProcessor()
    query = processor.process_stream()
    query.awaitTermination()
