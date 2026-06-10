# src/streaming/consumer_stateful.py
"""
Stateful Stream Processing Consumer for Amastan Fraud Shield Guard

Implements multi-transaction pattern detection via stateful windowed aggregation.
This replaces the stateless filter approach with actual behavioral analysis:
- "Has this card been used in 3 different governorates in the last hour?"
- "Is this user performing smurfing (1400-1500 TND repeatedly)?"
- "What is this user's baseline behavior vs current transaction?"

Architecture:
1. Kafka ingest -> quality gates (stateless)
2. Stateful aggregation via Spark Structured Streaming groupBy/agg windows
3. Per-user state tracking via mapInPandas (PySpark equivalent of mapGroupsWithState)
4. ML inference on enriched features
5. Alert dispatch with SAR generation
"""

import logging
import os
import threading
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

hadoop_home = os.getenv("HADOOP_HOME")
if hadoop_home:
    os.environ["HADOOP_HOME"] = hadoop_home

import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import approx_count_distinct, avg, col, count, expr, from_json, lit
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min
from pyspark.sql.functions import stddev, to_timestamp, when, window
from pyspark.sql.types import DoubleType, IntegerType

from shared.pii_masking import hash_pii
from shared.quality_gates import apply_d17_rule, validate_transaction_quality
from shared.schemas import TRANSACTION_SPARK_SCHEMA
from shared.utils import get_sqlite_connection, log_failed_alert, make_authenticated_request, retry_failed_alerts

schema = TRANSACTION_SPARK_SCHEMA

logger = logging.getLogger(__name__)


class StatefulFraudProcessor:
    """
    Stateful fraud processor with multi-transaction pattern detection.

    Features:
    - Per-user state tracking across time windows
    - Velocity analysis (transactions per window)
    - Geographic anomaly detection (impossible travel)
    - Smurfing pattern detection (repeated amounts in threshold range)
    - Baseline deviation analysis (current tx vs user history)
    - D17/e-wallet specific rule enforcement
    """

    def __init__(self, kafka_bootstrap=None):
        self.spark = (
            SparkSession.builder.appName("Tunisia-Fraud-Stateful-Silver")
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1")
            .config("spark.sql.streaming.checkpointLocation", "./tmp/checkpoint_stateful")
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
            .getOrCreate()
        )

        self.spark.sparkContext.setLogLevel("WARN")

        if kafka_bootstrap is None:
            kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
        self.kafka_bootstrap = kafka_bootstrap
        self._feedback_db_path = "./data/feedback.db"

        # Load XGBoost model for inference
        try:
            from xgboost.spark import SparkXGBClassifierModel

            model_path = self._get_champion_model_path()
            if model_path:
                self.ml_model = SparkXGBClassifierModel.load(model_path)
                logger.info(f"Champion model loaded: {model_path}")
            else:
                self.ml_model = None
                logger.info("No champion model found, using rule-based scoring")
        except Exception as e:
            self.ml_model = None
            logger.warning(f"ML model unavailable: {e}")

    def start_dlq_retry_worker(self):
        """Start the DLQ retry background thread."""
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
                    logging.exception("DLQ retry worker error")
                self._dlq_retry_stop.wait(interval)

        self._dlq_retry_thread = threading.Thread(target=retry_loop, name="dlq-retry-worker", daemon=True)
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

    def _ensure_model_registry_table(self):
        conn = get_sqlite_connection(self._feedback_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
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
        """
        )
        conn.commit()
        conn.close()

    def _get_champion_model_path(self):
        try:
            self._ensure_model_registry_table()
            conn = get_sqlite_connection(self._feedback_db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT model_path FROM model_registry
                WHERE is_champion = 1 ORDER BY promoted_at DESC LIMIT 1
            """
            )
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception:
            return None

    def _load_sampling_config(self):
        random_sample_rate = self._parse_float_env("RANDOM_SAMPLE_RATE", 0.01)
        random_sample_max_prob = self._parse_float_env("RANDOM_SAMPLE_MAX_PROB", 0.1)
        return max(0.0, min(random_sample_rate, 1.0)), max(0.0, min(random_sample_max_prob, 1.0))

    def _load_alerting_config(self):
        max_workers = max(1, self._parse_int_env("THREAD_POOL_SIZE", 5))
        async_timeout = max(1, self._parse_int_env("ALERT_ASYNC_TIMEOUT_SECONDS", 15))
        return max_workers, async_timeout

    def _load_uncertainty_zone(self):
        raw_zone = os.getenv("UNCERTAINTY_ZONE", "0.4,0.6")
        parts = [p.strip() for p in raw_zone.split(",") if p.strip()]
        if len(parts) != 2:
            return 0.4, 0.6
        try:
            low, high = float(parts[0]), float(parts[1])
        except ValueError:
            return 0.4, 0.6
        return max(0.0, min(low, 1.0)), max(0.0, min(high, 1.0))

    # ==================== Stateful Pattern Detection ====================

    def _enrich_with_stateful_features(self, validated_df):
        """
        Add stateful features by joining individual transactions with
        aggregated window state. This is the PySpark equivalent of
        mapGroupsWithState for stateful pattern detection.

        Features added:
        - v_count: Number of transactions by this user in the window
        - g_dist: Number of distinct governorates in the window
        - avg_amount: Average transaction amount in the window
        - amount_stddev: Amount variability (smurfing detection)
        - max_amount: Maximum amount in the window
        - payment_methods: List of payment methods used
        - time_since_first_tx: Time span of activity in window
        - is_repeat_governorate: Whether user is in same location as before
        """
        # Add event time and watermark
        enriched = validated_df.withColumn("event_time", to_timestamp(col("timestamp"))).withWatermark(
            "event_time", "10 minutes"
        )

        # Apply D17 rule
        enriched_with_d17 = apply_d17_rule(enriched)

        # Stateful windowed aggregation (5-min window, 1-min slide)
        # This creates per-user state that persists across micro-batches
        windowed_state = enriched_with_d17.groupBy(
            window(col("event_time"), "5 minutes", "1 minute"), col("user_id")
        ).agg(
            count("transaction_id").alias("v_count"),
            approx_count_distinct("governorate").alias("g_dist"),
            avg("amount_tnd").alias("avg_amount"),
            stddev("amount_tnd").alias("amount_stddev"),
            spark_max("amount_tnd").alias("max_amount"),
            spark_min("amount_tnd").alias("min_amount"),
            approx_count_distinct("payment_method").alias("payment_method_diversity"),
            expr("sum(case when payment_method = 'Flouci' then 1 else 0 end)").alias("flouci_count"),
            expr("sum(case when amount_tnd between 1400 and 1500 then 1 else 0 end)").alias("smurfing_count"),
            expr("sum(case when payment_method = 'Flouci' and amount_tnd > 2000 then 1 else 0 end)").alias(
                "d17_trigger_count"
            ),
        )

        # Join window state back to individual transactions
        # This gives each transaction the context of its user's window state
        stateful_enriched = enriched_with_d17.join(windowed_state, on=["user_id"], how="left")

        # Calculate temporal features
        stateful_enriched = (
            stateful_enriched.withColumn(
                "amount_in_smurfing_range", when(col("amount_tnd").between(1400, 1500), 1).otherwise(0)
            )
            .withColumn("is_high_velocity", when(col("v_count") > 5, 1).otherwise(0))
            .withColumn(
                "amount_coefficient_of_variation",
                when(
                    col("amount_stddev").isNotNull() & (col("avg_amount") > 0), col("amount_stddev") / col("avg_amount")
                ).otherwise(0.0),
            )
            .withColumn("is_multi_location", when(col("g_dist") > 1, 1).otherwise(0))
        )

        # PII hashing for compliance
        stateful_enriched = stateful_enriched.withColumn(
            "user_id_hashed", expr("sha2(user_id, 256)")  # SHA-256 hash of user_id
        )

        return stateful_enriched

    def _apply_weighted_risk_scoring(self, stateful_df):
        """
        Apply weighted risk scoring using stateful features.
        Uses dynamic rules engine if available, falls back to defaults.
        """
        from shared.rules_engine import get_rules_engine

        try:
            engine = get_rules_engine()
            weights = engine.get_risk_weights()
            high_value_threshold = engine.get_high_value_threshold()
        except Exception:
            # Fallback to compiled defaults
            from shared.risk_config import RISK_WEIGHTS
            from shared.rules_engine import DEFAULT_HIGH_VALUE_THRESHOLD

            weights = RISK_WEIGHTS
            high_value_threshold = DEFAULT_HIGH_VALUE_THRESHOLD

        scored = (
            stateful_df.withColumn("velocity_risk", when(col("v_count") > 3, lit(1.0)).otherwise(lit(0.0)))
            .withColumn("travel_risk", when(col("g_dist") > 1, lit(1.0)).otherwise(lit(0.0)))
            .withColumn(
                "high_value_risk", when(col("avg_amount") > lit(high_value_threshold), lit(1.0)).otherwise(lit(0.0))
            )
            .withColumn("d17_risk", when(col("d17_trigger_count") > 0, lit(1.0)).otherwise(lit(0.0)))
            .withColumn("smurfing_risk", when(col("smurfing_count") > 2, lit(1.0)).otherwise(lit(0.0)))
            .withColumn(
                "risk_score",
                (col("velocity_risk") * lit(weights["velocity"]))
                + (col("travel_risk") * lit(weights["travel"]))
                + (col("high_value_risk") * lit(weights["high_value"]))
                + (col("d17_risk") * lit(weights["d17_limit"]))
                + (col("smurfing_risk") * 0.15),  # Additional smurfing weight
            )
        )

        return scored

    def _apply_ml_inference(self, features_df):
        """Apply ML model inference on stateful features."""
        if not self.ml_model:
            return features_df.withColumn("ml_prediction", lit(-1)).withColumn("ml_probability", lit(0.0))

        from pyspark.ml.feature import VectorAssembler

        assembler = VectorAssembler(
            inputCols=[
                "v_count",
                "g_dist",
                "avg_amount",
                "is_smurfing",
                "high_velocity_flag",
                "velocity_risk",
                "travel_risk",
                "high_value_risk",
                "d17_risk",
                "risk_score",
                "smurfing_risk",
                "amount_coefficient_of_variation",
            ],
            outputCol="features",
            handleInvalid="skip",
        )

        assembled = assembler.transform(features_df)
        predictions = self.ml_model.transform(assembled)

        return predictions.withColumnRenamed("prediction", "ml_prediction").withColumnRenamed(
            "probability", "ml_probability"
        )

    def _count_new_feedback_since_promotion(self):
        try:
            self._ensure_model_registry_table()
            conn = get_sqlite_connection(self._feedback_db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT promoted_at FROM model_registry
                WHERE is_champion = 1 ORDER BY promoted_at DESC LIMIT 1
            """
            )
            row = cursor.fetchone()
            if row and row[0]:
                cursor.execute(
                    "SELECT COUNT(*) FROM feedback_labels WHERE analyst_label IS NOT NULL AND timestamp > ?",
                    (row[0],),
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM feedback_labels WHERE analyst_label IS NOT NULL")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def _send_alert_async(self, row, sar_gen, alert_type="high_risk", generate_sar=True):
        """Send alert to command center API asynchronously."""
        try:
            row_dict = row.asDict()
            ml_probability = float(row_dict.get("ml_probability", 0.0) or 0.0)

            event_timestamp_str = str(row_dict.get("timestamp", ""))
            ingestion_latency = 0.0
            if event_timestamp_str:
                try:
                    event_time = datetime.fromisoformat(event_timestamp_str.replace("Z", "+00:00"))
                    processing_time = datetime.now()
                    ingestion_latency = (processing_time - event_time).total_seconds()
                except Exception:
                    pass

            report = None
            if generate_sar and sar_gen is not None:
                report = sar_gen.generate_report(row_dict, ml_probability)
                report_path = sar_gen.save_report(row_dict, report, ml_probability)
                logger.info(f"SAR saved: {report_path}")

            # Hash user_id for PII compliance
            user_id_hashed = hash_pii(str(row_dict.get("user_id", "")))

            alert_payload = {
                "transaction_id": str(row_dict.get("transaction_id", "unknown")),
                "user_id": str(row_dict.get("user_id", "unknown")),
                "user_id_hashed": user_id_hashed,
                "amount_tnd": float(row_dict.get("amount_tnd", 0.0) or 0.0),
                "governorate": str(row_dict.get("governorate", "unknown")),
                "payment_method": str(row_dict.get("payment_method", "unknown")),
                "branch_id": str(row_dict.get("branch_id", "unknown")),
                "timestamp": event_timestamp_str,
                "ml_probability": ml_probability,
                "risk_score": float(row_dict.get("risk_score", 0.0) or 0.0),
                "v_count": int(row_dict.get("v_count", 0) or 0),
                "g_dist": int(row_dict.get("g_dist", 0) or 0),
                "smurfing_count": int(row_dict.get("smurfing_count", 0) or 0),
                "sar_report": report,
                "alert_type": alert_type,
                "ingestion_latency": ingestion_latency,
            }

            try:
                start_time = time.time()
                api_response = make_authenticated_request("POST", "/alerts/add/", payload=alert_payload, timeout=5)
                api_call_duration = time.time() - start_time

                if api_response and api_response.status_code == 200:
                    logger.info(
                        f"Alert sent (latency: {ingestion_latency:.2f}s, API: {api_call_duration:.2f}s) "
                        f"for tx {row_dict.get('transaction_id')} ({alert_type})"
                    )
                else:
                    error_msg = f"{api_response.status_code} - {api_response.text}" if api_response else "No response"
                    error_code = str(api_response.status_code) if api_response else "NO_RESPONSE"
                    logger.warning(f"Alert failed: {error_msg}")
                    log_failed_alert(row_dict, alert_payload, error_code, error_msg)
            except Exception as api_error:
                logger.warning(f"API error: {api_error}")
                log_failed_alert(row_dict, alert_payload, "CONNECTION_ERROR", str(api_error))

        except Exception as e:
            try:
                row_dict = row.asDict()
            except Exception:
                row_dict = {"transaction_id": "unknown"}
            logger.error(f"Error processing tx {row_dict.get('transaction_id')}: {e}")
            log_failed_alert(row_dict, {}, "PROCESSING_ERROR", str(e))

    def _process_batch(self, batch_df, epoch_id):
        """Process a micro-batch: route alerts based on ML probability and sampling config."""
        random_sample_rate, random_sample_max_prob = self._load_sampling_config()
        uncertainty_low, uncertainty_high = self._load_uncertainty_zone()

        high_risk_rows = batch_df.filter(col("ml_probability") > 0.85).collect()
        sampled_low_risk_rows = []
        uncertainty_rows = []

        if random_sample_rate > 0:
            low_risk_df = batch_df.filter(col("ml_probability") < random_sample_max_prob)
            if random_sample_rate < 1:
                low_risk_df = low_risk_df.sample(withReplacement=False, fraction=random_sample_rate)
            sampled_low_risk_rows = low_risk_df.collect()

        uncertainty_rows = batch_df.filter(
            (col("ml_probability") >= uncertainty_low) & (col("ml_probability") <= uncertainty_high)
        ).collect()

        if not high_risk_rows and not sampled_low_risk_rows and not uncertainty_rows:
            return

        from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

        from rag_engine.sar_generator import SARGenerator

        sar_gen = SARGenerator() if high_risk_rows else None
        max_workers, async_timeout = self._load_alerting_config()

        logger.info(
            f"Epoch {epoch_id}: {len(high_risk_rows)} high-risk, "
            f"{len(sampled_low_risk_rows)} random, {len(uncertainty_rows)} uncertainty"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {}
            for row in high_risk_rows:
                future = executor.submit(self._send_alert_async, row, sar_gen, "high_risk", True)
                future_to_row[future] = row
            for row in sampled_low_risk_rows:
                future = executor.submit(self._send_alert_async, row, None, "random_sample", False)
                future_to_row[future] = row
            for row in uncertainty_rows:
                future = executor.submit(self._send_alert_async, row, None, "uncertainty_sample", False)
                future_to_row[future] = row

            try:
                for future in as_completed(future_to_row, timeout=async_timeout):
                    try:
                        future.result()
                    except Exception as e:
                        row = future_to_row[future]
                        try:
                            tx_id = row.asDict().get("transaction_id", "unknown")
                        except Exception:
                            tx_id = "unknown"
                        logger.error(f"Async alert error for tx {tx_id}: {e}")
            except TimeoutError:
                logger.warning("Alert processing timed out")

    def _check_and_trigger_retraining(self, batch_df, epoch_id):
        """
        Process a micro-batch with idempotency enforcement.
        Each transaction is checked against the dedup cache before processing.
        """
        from src.shared.idempotency import get_dedup_cache

        dedup = get_dedup_cache()
        random_sample_rate, random_sample_max_prob = self._load_sampling_config()
        uncertainty_low, uncertainty_high = self._load_uncertainty_zone()

        # Filter out already-processed transactions (idempotency)
        original_count = batch_df.count()

        def dedup_filter(rows):
            """Filter out duplicate transactions within a batch."""
            unique_rows = []
            for row in rows:
                tx_id = row.asDict().get("transaction_id", "")
                if not dedup.is_duplicate(tx_id):
                    dedup.mark_processed(tx_id)
                    unique_rows.append(row)
            return unique_rows

        # Collect and dedup
        all_rows = batch_df.collect()
        unique_rows = dedup_filter(all_rows)

        duplicates_skipped = original_count - len(unique_rows)
        if duplicates_skipped > 0:
            logger.info(f"Epoch {epoch_id}: Skipped {duplicates_skipped} duplicate transactions")

        if not unique_rows:
            return

        # Convert back to DataFrame for filtering
        unique_df = batch_df.sparkSession.createDataFrame(
            [r.asDict() for r in unique_rows],
            schema=batch_df.schema,
        )

        # Route alerts based on ML probability
        high_risk_rows = unique_df.filter(col("ml_probability") > 0.85).collect()
        sampled_low_risk_rows = []
        uncertainty_rows = []

        if random_sample_rate > 0:
            low_risk_df = unique_df.filter(col("ml_probability") < random_sample_max_prob)
            if random_sample_rate < 1:
                low_risk_df = low_risk_df.sample(withReplacement=False, fraction=random_sample_rate)
            sampled_low_risk_rows = low_risk_df.collect()

        uncertainty_rows = unique_df.filter(
            (col("ml_probability") >= uncertainty_low) & (col("ml_probability") <= uncertainty_high)
        ).collect()

        if not high_risk_rows and not sampled_low_risk_rows and not uncertainty_rows:
            return

        # Process alerts (with shadow model comparison if active)
        from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

        from ml.shadow_model import ShadowModelManager
        from rag_engine.sar_generator import SARGenerator

        # Shadow model scoring
        shadow = ShadowModelManager()
        shadow_df = None
        if shadow.has_shadow_model():
            _, shadow_df = shadow.score_with_both(unique_df)
            if shadow_df is not None:
                # Log comparisons
                shadow_rows = shadow_df.select("transaction_id", "ml_probability", "shadow_probability").collect()
                for srow in shadow_rows:
                    shadow.record_shadow_comparison(
                        srow["transaction_id"],
                        float(srow["ml_probability"] or 0),
                        float(srow["shadow_probability"] or 0),
                    )

        sar_gen = SARGenerator() if high_risk_rows else None
        max_workers, async_timeout = self._load_alerting_config()

        logger.info(
            f"Epoch {epoch_id}: {len(high_risk_rows)} high-risk, "
            f"{len(sampled_low_risk_rows)} random, {len(uncertainty_rows)} uncertainty"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {}
            for row in high_risk_rows:
                future = executor.submit(self._send_alert_async, row, sar_gen, "high_risk", True)
                future_to_row[future] = row
            for row in sampled_low_risk_rows:
                future = executor.submit(self._send_alert_async, row, None, "random_sample", False)
                future_to_row[future] = row
            for row in uncertainty_rows:
                future = executor.submit(self._send_alert_async, row, None, "uncertainty_sample", False)
                future_to_row[future] = row

            try:
                for future in as_completed(future_to_row, timeout=async_timeout):
                    try:
                        future.result()
                    except Exception as e:
                        row = future_to_row[future]
                        try:
                            tx_id = row.asDict().get("transaction_id", "unknown")
                        except Exception:
                            tx_id = "unknown"
                        logger.error(f"Async alert error for tx {tx_id}: {e}")
            except TimeoutError:
                logger.warning("Alert processing timed out")

        # Check for retraining trigger
        try:
            new_feedback_count = self._count_new_feedback_since_promotion()
            threshold = max(1, self._parse_int_env("RETRAIN_FEEDBACK_THRESHOLD", 100))
            if new_feedback_count < threshold:
                return

            logger.info(f"Retraining triggered: {new_feedback_count} new feedback records")
            retrain_response = make_authenticated_request("POST", "/retrain-model/", timeout=10)

            if retrain_response and retrain_response.status_code == 200:
                logger.info("Model retraining triggered successfully")
            else:
                status = retrain_response.status_code if retrain_response else "NO_RESPONSE"
                logger.warning(f"Retraining trigger failed: {status}")
        except Exception as e:
            logger.error(f"Retraining check error: {e}")

    def process_stream(self):
        """
        Main stream processing pipeline with idempotency:
        1. Kafka ingest (Bronze)
        2. Quality gates (stateless validation)
        3. Stateful enrichment (windowed aggregation per user)
        4. Weighted risk scoring (rules engine)
        5. ML inference (XGBoost + shadow model)
        6. Idempotent alert dispatch (dedup cache)
        7. Parquet persistence (Silver)
        """
        # 1. Bronze: Ingest from Kafka
        raw_stream = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap)
            .option("subscribe", "tunisian_transactions")
            .option("failOnDataLoss", "false")
            .load()
        )

        # Deserialize JSON
        json_df = (
            raw_stream.selectExpr("CAST(value AS STRING)")
            .select(from_json(col("value"), schema).alias("data"))
            .select("data.*")
        )

        # 2. Quality gates
        validated_df = validate_transaction_quality(json_df)

        # 3. Stateful enrichment (the key differentiator)
        stateful_df = self._enrich_with_stateful_features(validated_df)

        # 4. Risk scoring
        scored_df = self._apply_weighted_risk_scoring(stateful_df)

        # Ensure consistent feature columns for ML
        feature_cols = [
            "v_count",
            "g_dist",
            "avg_amount",
            "is_smurfing",
            "high_velocity_flag",
            "velocity_risk",
            "travel_risk",
            "high_value_risk",
            "d17_risk",
            "risk_score",
            "smurfing_risk",
            "amount_coefficient_of_variation",
        ]

        for col_name in feature_cols:
            if col_name not in scored_df.columns:
                if col_name in ("avg_amount", "amount_stddev", "amount_coefficient_of_variation"):
                    scored_df = scored_df.withColumn(col_name, lit(0.0).cast(DoubleType()))
                else:
                    scored_df = scored_df.withColumn(col_name, lit(0).cast(IntegerType()))

        # 5. ML inference
        final_df = self._apply_ml_inference(scored_df)

        # 6 & 7. Idempotent alert dispatch + persistence
        query = (
            final_df.writeStream.format("parquet")
            .outputMode("append")
            .option("path", "./data/parquet/silver_fraud_alerts")
            .option("checkpointLocation", "./tmp/checkpoint_stateful/silver_fraud")
            .foreachBatch(self._check_and_trigger_retraining)
            .start()
        )

        self.start_dlq_retry_worker()
        return query


if __name__ == "__main__":
    processor = StatefulFraudProcessor()
    query = processor.process_stream()
    query.awaitTermination()
