# src/streaming/consumer.py
import os
import sys
from pathlib import Path

# Explicitly set HADOOP_HOME for the subprocess (critical for Windows)
os.environ['HADOOP_HOME'] = r'C:\hadoop-3.4.2'

# Ensure 'src' is in path for schema import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, count, approx_count_distinct, current_timestamp, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType
from shared.schemas import Transaction

# Define Spark Schema matching Pydantic Transaction model
schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("amount_tnd", DoubleType(), True),
    StructField("governorate", StringType(), True),
    StructField("payment_method", StringType(), True),
    StructField("fraud_seed", BooleanType(), True)
])

def start_streaming():
    # Initialize Spark with compatible Kafka packages for Spark 4.1.1
    # Use a temporary directory for checkpoint that doesn't rely on Hadoop native libraries
    spark = SparkSession.builder \
        .appName("TunisianFraudDetection-Ingestion") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1") \
        .config("spark.sql.streaming.checkpointLocation", "./tmp/checkpoint") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Read from Kafka topic verified on Day 1
    raw_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "tunisian_transactions") \
        .option("startingOffsets", "latest") \
        .load()

    # Deserialize JSON value
    json_df = raw_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # 1. Add a proper timestamp column and define a Watermark (handling late data)
    # We allow data to be up to 10 minutes late
    enriched_df = json_df.withColumn("event_time", to_timestamp(col("timestamp"))) \
                         .withWatermark("event_time", "10 minutes")

    # 2. Stateful Aggregation: Detection of Velocity Attacks
    # Pattern: Count transactions per user in a 5-minute sliding window, updating every 1 minute
    velocity_checks = enriched_df.groupBy(
        window(col("event_time"), "5 minutes", "1 minute"),
        col("user_id")
    ).agg(
        count("transaction_id").alias("tx_count"),
        approx_count_distinct("governorate").alias("distinct_governorates")
    )

    # 3. Flagging Logic: Defining the "Fraud Alert"
    # Fraud if > 3 transactions in 5 mins OR > 1 city in 5 mins (Impossible Travel)
    alerts_df = velocity_checks.filter(
        (col("tx_count") > 3) | (col("distinct_governorates") > 1)
    ).select(
        col("window.start").alias("start_time"),
        col("window.end").alias("end_time"),
        "user_id", "tx_count", "distinct_governorates"
    )

    # 4. Console Sink for Alerts
    query = alerts_df.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    start_streaming()