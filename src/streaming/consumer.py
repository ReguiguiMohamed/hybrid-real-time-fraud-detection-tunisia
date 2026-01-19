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
from shared.schemas import Transaction, TRANSACTION_SPARK_SCHEMA
from shared.risk_config import RISK_WEIGHTS, CBDC_PILOT_GOVERNORATES, D17_SOFT_LIMIT, D17_VELOCITY_CAP
from shared.quality_gates import validate_transaction_quality, apply_d17_rule

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

        # Apply ML inference if model is available
        if self.ml_model:
            # Prepare features for ML model
            features_df = scored.withColumn("is_smurfing", when(col("avg_amount").between(1400, 1500), 1).otherwise(0)) \
                                .withColumn("high_velocity_flag", when(col("v_count") > D17_VELOCITY_CAP, 1).otherwise(0))

            # Apply ML model for prediction
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

            # Trigger SAR generation for high-confidence fraud predictions
            from pyspark.sql.functions import udf
            from pyspark.sql.types import StringType
            from rag_engine.sar_generator import SARGenerator

            # Initialize SAR generator
            sar_gen = SARGenerator()

            # Function to trigger SAR generation
            def generate_sar_if_needed(row_dict, ml_prob_val):
                if ml_prob_val > 0.85:  # High confidence threshold
                    try:
                        report = sar_gen.generate_report(row_dict, ml_prob_val)
                        report_path = sar_gen.save_report(row_dict, report, ml_prob_val)
                        print(f"SAR generated and saved to: {report_path}")
                        return report_path
                    except Exception as e:
                        print(f"Error generating SAR: {e}")
                        return None
                return None

            # Create UDF for SAR generation
            sar_udf = udf(lambda row_dict, prob: generate_sar_if_needed(row_dict, prob), StringType())

        else:
            # Fallback to rule-based scoring
            final_df = scored.withColumn("ml_prediction", lit(-1)) \
                            .withColumn("ml_probability", lit(0.0))

        # 4. Persistence: Using Parquet for streaming (Delta Lake for batch operations)
        # Due to compatibility issues between Spark 4.1.1 and Delta Lake 4.0.1 for streaming sinks
        query = final_df.writeStream \
            .format("parquet") \
            .outputMode("append") \
            .option("path", "./data/parquet/silver_fraud_alerts") \
            .option("checkpointLocation", "./tmp/checkpoint/silver_fraud") \
            .start()

        return query

if __name__ == "__main__":
    processor = FraudProcessor()
    query = processor.process_stream()
    query.awaitTermination()