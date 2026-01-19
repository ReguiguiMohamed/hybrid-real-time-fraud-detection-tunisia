# src/streaming/consumer.py
import os
import logging
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
import sqlite3
import os
from datetime import datetime

# Use the schema from the shared module to ensure consistency
schema = TRANSACTION_SPARK_SCHEMA

class FraudProcessor:
    def __init__(self, kafka_bootstrap="localhost:9092"):
        # Initializing with Kafka support (Delta Lake config removed to avoid streaming conflicts)
        self.spark = SparkSession.builder \
            .appName("Tunisia-Fraud-Silver-Layer") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1") \
            .config("spark.sql.streaming.checkpointLocation", "./tmp/checkpoint") \
            .getOrCreate()
        self.kafka_bootstrap = kafka_bootstrap

        # Load XGBoost model for real-time inference
        try:
            from xgboost.spark import SparkXGBClassifierModel
            self.ml_model = SparkXGBClassifierModel.load("models/fraud_xgb_v1")
            print("✅ XGBoost Model loaded for Real-Time Inference")
        except Exception as e:
            print(f"⚠️ Fallback to Rule-Based Scoring. Model not available: {e}")
            self.ml_model = None

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

        # For performance, we'll use foreachBatch to handle SAR generation and alerting asynchronously
        # This avoids blocking the streaming pipeline with HTTP requests to Ollama
        def process_batch(batch_df, epoch_id):
            # Filter for high-confidence fraud predictions
            high_risk_df = batch_df.filter(col("ml_probability") > 0.85).collect()

            if high_risk_df:
                from rag_engine.sar_generator import SARGenerator
                import requests
                import json
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading

                sar_gen = SARGenerator()

                # Create a thread pool with configurable max workers
                max_workers = int(os.getenv("THREAD_POOL_SIZE", "10"))

                # Submit tasks to thread pool
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_row = {
                        executor.submit(send_alert_async, row, sar_gen): row
                        for row in high_risk_df
                    }

                    # Wait for all tasks to complete with timeout
                    for future in as_completed(future_to_row, timeout=30):  # 30 second timeout for all tasks
                        try:
                            future.result()
                        except Exception as e:
                            row = future_to_row[future]
                            print(f"Error in async alert processing for transaction {row['transaction_id']}: {e}")

        def send_alert_async(row, sar_gen):
            """Function to send alerts asynchronously"""
            try:
                # Convert row to dictionary for SAR generation
                row_dict = row.asDict()
                report = sar_gen.generate_report(row_dict, float(row.ml_probability))

                # Save the SAR report
                report_path = sar_gen.save_report(row_dict, report, float(row.ml_probability))
                print(f"SAR generated and saved to: {report_path}")

                # Send alert to the command center API
                alert_payload = {
                    "transaction_id": str(row_dict.get('transaction_id', 'unknown')),
                    "user_id": str(row_dict.get('user_id', 'unknown')),
                    "amount_tnd": float(row_dict.get('amount_tnd', 0.0)),
                    "governorate": str(row_dict.get('governorate', 'unknown')),
                    "payment_method": str(row_dict.get('payment_method', 'unknown')),
                    "timestamp": str(row_dict.get('timestamp', '')),
                    "ml_probability": float(row.ml_probability),
                    "sar_report": report
                }

                try:
                    # Get API URL from environment variable, default to localhost
                    api_url = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
                    api_token = os.getenv("COMMAND_CENTER_API_TOKEN")

                    headers = {"Content-Type": "application/json"}
                    if api_token:
                        headers["Authorization"] = f"Bearer {api_token}"

                    api_response = requests.post(
                        f"{api_url}/alerts/add/",
                        json=alert_payload,
                        headers=headers,
                        timeout=5  # 5 second timeout to avoid blocking
                    )

                    if api_response.status_code == 200:
                        print(f"Alert sent to command center for transaction: {row_dict.get('transaction_id')}")
                    else:
                        print(f"Failed to send alert to command center: {api_response.status_code} - {api_response.text}")
                        # Log to dead letter queue
                        log_failed_alert(row_dict, alert_payload, str(api_response.status_code), api_response.text)
                except requests.exceptions.RequestException as api_error:
                    print(f"API connection error when sending alert: {api_error}")
                    # Log to dead letter queue
                    log_failed_alert(row_dict, alert_payload, "CONNECTION_ERROR", str(api_error))

            except Exception as e:
                print(f"Error processing high-risk transaction {row.get('transaction_id', 'unknown')}: {e}")
                # Log to dead letter queue
                row_dict = row.asDict() if 'row' in locals() else {"transaction_id": "unknown"}
                log_failed_alert(row_dict, {}, "PROCESSING_ERROR", str(e))

        # Periodically check if we should trigger model retraining based on feedback
        def check_and_trigger_retraining(batch_df, epoch_id):
            # Call the process_batch function first
            process_batch(batch_df, epoch_id)

            # Every few batches, check if we should trigger retraining
            # We'll use a simple counter to trigger retraining every N batches
            # In a real system, this could be based on time intervals or feedback accumulation
            global batch_counter
            if 'batch_counter' not in globals():
                batch_counter = 0
            batch_counter += 1

            # Trigger retraining every 10 batches (this is configurable)
            if batch_counter % 10 == 0:
                try:
                    # Check if we have sufficient feedback to warrant retraining
                    import sqlite3
                    conn = sqlite3.connect("./data/feedback.db")
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM feedback_labels WHERE analyst_label IS NOT NULL")
                    feedback_count = cursor.fetchone()[0]
                    conn.close()

                    # Only trigger retraining if we have enough feedback
                    if feedback_count >= 50:  # Minimum threshold for meaningful retraining
                        print(f"Triggering model retraining based on {feedback_count} feedback records")

                        # Call the retraining endpoint with proper authentication
                        api_url = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
                        api_token = os.getenv("COMMAND_CENTER_API_TOKEN")

                        headers = {"Content-Type": "application/json"}
                        if api_token:
                            headers["Authorization"] = f"Bearer {api_token}"

                        retrain_response = requests.post(
                            f"{api_url}/retrain-model/",
                            headers=headers,
                            timeout=10  # 10 second timeout for retraining trigger
                        )

                        if retrain_response.status_code == 200:
                            print("✅ Model retraining triggered successfully")
                        else:
                            print(f"Failed to trigger model retraining: {retrain_response.status_code} - {retrain_response.text}")

                except Exception as e:
                    print(f"Error checking feedback for retraining: {e}")

        # 4. Persistence: Using Parquet for streaming (Delta Lake for batch operations)
        # Due to compatibility issues between Spark 4.1.1 and Delta Lake 4.0.1 for streaming sinks
        query = final_df.writeStream \
            .format("parquet") \
            .outputMode("append") \
            .option("path", "./data/parquet/silver_fraud_alerts") \
            .option("checkpointLocation", "./tmp/checkpoint/silver_fraud") \
            .foreachBatch(check_and_trigger_retraining) \
            .start()

        def log_failed_alert(transaction_data, alert_payload, error_code, error_message):
            """Log failed alerts to a dead letter queue for later processing"""
            try:
                # Create dead letter database if it doesn't exist
                db_path = "./data/dead_letter_queue.db"
                os.makedirs(os.path.dirname(db_path), exist_ok=True)

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS failed_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transaction_id TEXT,
                        user_id TEXT,
                        amount_tnd REAL,
                        governorate TEXT,
                        payment_method TEXT,
                        timestamp TEXT,
                        ml_probability REAL,
                        error_code TEXT,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Insert failed alert
                cursor.execute("""
                    INSERT INTO failed_alerts
                    (transaction_id, user_id, amount_tnd, governorate, payment_method,
                     timestamp, ml_probability, error_code, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transaction_data.get('transaction_id'),
                    transaction_data.get('user_id'),
                    transaction_data.get('amount_tnd'),
                    transaction_data.get('governorate'),
                    transaction_data.get('payment_method'),
                    transaction_data.get('timestamp'),
                    transaction_data.get('ml_probability', 0.0),
                    error_code,
                    error_message
                ))

                conn.commit()
                conn.close()

                print(f"Failed alert logged to dead letter queue: {transaction_data.get('transaction_id')}")
            except Exception as e:
                print(f"Error logging failed alert to dead letter queue: {e}")

        return query

if __name__ == "__main__":
    processor = FraudProcessor()
    query = processor.process_stream()
    query.awaitTermination()