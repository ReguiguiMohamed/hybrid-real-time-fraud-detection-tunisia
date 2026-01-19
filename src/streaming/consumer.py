# src/streaming/consumer.py
import os
import sys
from pathlib import Path

# Explicitly set HADOOP_HOME for the subprocess (critical for Windows)
os.environ['HADOOP_HOME'] = r'C:\hadoop-3.4.2'

# Ensure 'src' is in path for schema import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
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

    # Console Sink for Verification
    query = json_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    start_streaming()