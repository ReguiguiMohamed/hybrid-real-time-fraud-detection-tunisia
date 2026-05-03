# src/streaming/consumer.py
import os
import logging
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set HADOOP_HOME from environment variable (required for PySpark on Windows)
hadoop_home = os.getenv('HADOOP_HOME')
if hadoop_home:
    os.environ['HADOOP_HOME'] = hadoop_home

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, count, approx_count_distinct, when, lit, to_timestamp, expr
from pyspark.sql.types import DoubleType
from shared.schemas import Transaction, TRANSACTION_SPARK_SCHEMA
from shared.risk_config import RISK_WEIGHTS, CBDC_PILOT_GOVERNORATES, D17_SOFT_LIMIT, D17_VELOCITY_CAP
from shared.rules_engine import get_rules_engine
from shared.quality_gates import (
    validate_transaction_quality,
    apply_d17_rule,
    apply_tunicheque_rules,
    apply_ttn_rules,
    apply_fcy_rules,
    apply_device_behavior_rules,
)
from shared.utils import make_authenticated_request, log_failed_alert, retry_failed_alerts, get_sqlite_connection
from compliance.pkyc import PKYCPublisher
from compliance.sanctions import SanctionsScreener
import time

# Use the schema from the shared module to ensure consistency
schema = TRANSACTION_SPARK_SCHEMA

MODEL_FEATURE_COLS = [
    "v_count",
    "g_dist",
    "avg_amount",
    "is_smurfing",
    "smurfing_velocity_flag",
    "high_velocity_flag",
]

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

        self._feedback_db_path = "./data/feedback.db"
        self._shap_explainer = None
        self._model_feature_cols = MODEL_FEATURE_COLS
        self._pkyc_publisher = PKYCPublisher(bootstrap_servers=self.kafka_bootstrap)
        self._sanctions_screener = SanctionsScreener()

        # Load XGBoost model for real-time inference
        try:
            model_path = self._get_champion_model_path()
            if model_path:
                self.ml_model = self._load_champion_model(model_path)
                print(f"Model loaded from registry: {model_path}")
            else:
                print("No champion model registered. Using rule-based scoring.")
                self.ml_model = None
        except Exception as e:
            print(f"Fallback to Rule-Based Scoring. Model not available: {e}")
            self.ml_model = None

    def _load_champion_model(self, model_path):
        """Load a registry model and initialise SHAP from its XGBoost booster."""
        from pyspark.ml import PipelineModel
        from xgboost.spark import SparkXGBClassifierModel

        try:
            model = PipelineModel.load(model_path)
        except Exception:
            model = SparkXGBClassifierModel.load(model_path)

        self._configure_shap(model)
        return model

    def _extract_xgb_stage(self, model):
        if hasattr(model, "stages"):
            for stage in model.stages:
                if hasattr(stage, "getInputCols"):
                    self._model_feature_cols = list(stage.getInputCols())
                if hasattr(stage, "get_booster"):
                    return stage
            return None
        return model if hasattr(model, "get_booster") else None

    def _configure_shap(self, model):
        try:
            import shap

            xgb_stage = self._extract_xgb_stage(model)
            if not xgb_stage:
                logging.warning("SHAP unavailable: no XGBoost stage found in model")
                return

            booster = xgb_stage.get_booster()
            self._shap_explainer = shap.TreeExplainer(booster)
        except Exception:
            logging.exception("SHAP initialisation failed; alerts will omit shap_top5")
            self._shap_explainer = None

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _extract_probability(cls, value):
        """Return fraud-class probability from Spark vectors or scalar scores."""
        if value is None:
            return 0.0
        try:
            if hasattr(value, "toArray"):
                values = value.toArray().tolist()
                return cls._as_float(values[-1] if values else 0.0)
            if isinstance(value, (list, tuple)):
                return cls._as_float(value[-1] if value else 0.0)
            return cls._as_float(value)
        except Exception:
            return 0.0

    def _compute_shap_top5(self, row_dict):
        if not self._shap_explainer:
            return []

        try:
            import numpy as np

            feature_values = [
                self._as_float(row_dict.get(feature_name))
                for feature_name in self._model_feature_cols
            ]
            sample = np.array([feature_values], dtype=float)
            shap_values = self._shap_explainer.shap_values(sample)

            if isinstance(shap_values, list):
                values = shap_values[-1][0]
            else:
                values = shap_values[0]
                if getattr(values, "ndim", 1) > 1:
                    values = values[:, -1]

            ranked = []
            total_abs_impact = sum(abs(float(impact)) for impact in values) or 0.0
            for feature_name, feature_value, impact in zip(self._model_feature_cols, feature_values, values):
                impact_value = float(impact)
                ranked.append({
                    "feature": feature_name,
                    "value": feature_value,
                    "impact": impact_value,
                    "abs_impact": abs(impact_value),
                    "direction": "increases_risk" if impact_value >= 0 else "decreases_risk",
                })

            ranked.sort(key=lambda item: item["abs_impact"], reverse=True)
            top5 = ranked[:5]
            covered_impact = sum(item["abs_impact"] for item in top5)
            confidence = covered_impact / total_abs_impact if total_abs_impact else 0.0
            for item in top5:
                item["confidence"] = round(confidence, 6)
            return top5
        except Exception:
            logging.exception("SHAP scoring failed for transaction %s", row_dict.get("transaction_id", "unknown"))
            return []

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

    def _ensure_model_registry_table(self):
        conn = get_sqlite_connection(self._feedback_db_path)
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
        conn.commit()
        conn.close()

    def _get_champion_model_path(self):
        try:
            self._ensure_model_registry_table()
            conn = get_sqlite_connection(self._feedback_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_path
                FROM model_registry
                WHERE is_champion = 1
                ORDER BY promoted_at DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception:
            logging.exception("Unable to read champion model from registry")
            return None

    def _load_uncertainty_zone(self):
        raw_zone = os.getenv("UNCERTAINTY_ZONE", "0.4,0.6")
        parts = [part.strip() for part in raw_zone.split(",") if part.strip()]
        if len(parts) != 2:
            return 0.4, 0.6
        try:
            low = float(parts[0])
            high = float(parts[1])
        except ValueError:
            return 0.4, 0.6
        low = max(0.0, min(low, 1.0))
        high = max(0.0, min(high, 1.0))
        if low > high:
            low, high = high, low
        return low, high

    def _apply_sanctions_screening(self, df):
        sanctioned_accounts = list(self._sanctions_screener.account_ids)
        if not sanctioned_accounts:
            return df.withColumn("sanctions_hit_flag", lit(False))

        return df.withColumn(
            "sanctions_hit_flag",
            col("sender_account").isin(sanctioned_accounts) |
            col("receiver_account").isin(sanctioned_accounts) |
            col("user_id").isin(sanctioned_accounts),
        )

    def _count_new_feedback_since_promotion(self):
        try:
            self._ensure_model_registry_table()
            conn = get_sqlite_connection(self._feedback_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT promoted_at
                FROM model_registry
                WHERE is_champion = 1
                ORDER BY promoted_at DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row and row[0]:
                cursor.execute(
                    "SELECT COUNT(*) FROM feedback_labels WHERE analyst_label IS NOT NULL AND timestamp > ?",
                    (row[0],)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM feedback_labels WHERE analyst_label IS NOT NULL")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            logging.exception("Unable to count new feedback for retraining trigger")
            return 0

    def _send_alert_async(self, row, sar_gen, alert_type="high_risk", generate_sar=True):
        """Send an alert to the command center API."""
        try:
            row_dict = row.asDict()
            ml_probability = self._extract_probability(row_dict.get("ml_probability", 0.0))
            if alert_type == "SANCTIONS_HIT":
                ml_probability = 1.0
                row_dict["sanctions_hit"] = 1
            shap_top5 = self._compute_shap_top5(row_dict)
            row_dict["shap_top5"] = shap_top5
            min_shap_confidence = self._parse_float_env("SHAP_MIN_CONFIDENCE_FOR_SAR", 0.0)
            shap_confidence = shap_top5[0].get("confidence", 0.0) if shap_top5 else 0.0

            # Calculate ingestion latency: time from event timestamp to processing time
            import datetime
            event_timestamp_str = str(row_dict.get("timestamp", ""))
            if event_timestamp_str:
                try:
                    # Parse the event timestamp
                    event_time = datetime.datetime.fromisoformat(event_timestamp_str.replace('Z', '+00:00'))
                    processing_time = datetime.datetime.now(datetime.timezone.utc)
                    ingestion_latency = (processing_time - event_time).total_seconds()
                except:
                    ingestion_latency = 0.0  # Default if parsing fails
            else:
                ingestion_latency = 0.0

            report = None
            if generate_sar and sar_gen is not None and shap_confidence >= min_shap_confidence:
                report = sar_gen.generate_report(row_dict, ml_probability)
                report_path = sar_gen.save_report(row_dict, report, ml_probability)
                print(f"SAR generated and saved to: {report_path}")
            elif generate_sar and sar_gen is not None:
                print(
                    "SAR generation skipped for transaction "
                    f"{row_dict.get('transaction_id')} due to SHAP confidence "
                    f"{shap_confidence:.3f} below {min_shap_confidence:.3f}"
                )

            alert_payload = {
                "transaction_id": str(row_dict.get("transaction_id", "unknown")),
                "user_id": str(row_dict.get("user_id", "unknown")),
                "amount_tnd": float(row_dict.get("amount_tnd", 0.0) or 0.0),
                "governorate": str(row_dict.get("governorate", "unknown")),
                "payment_method": str(row_dict.get("payment_method", "unknown")),
                "branch_id": str(row_dict.get("branch_id", "unknown")),
                "timestamp": event_timestamp_str,
                "ml_probability": ml_probability,
                "sar_report": report,
                "alert_type": alert_type,
                "shap_top5": shap_top5,
                "ingestion_latency": ingestion_latency  # Include latency in payload for monitoring
            }

            try:
                pkyc_event = self._pkyc_publisher.publish_for_transaction(row_dict, ml_probability)
                if pkyc_event:
                    print(
                        "pKYC trigger published for transaction "
                        f"{row_dict.get('transaction_id')}: {pkyc_event.trigger_reason}"
                    )

                start_time = time.time()
                api_response = make_authenticated_request(
                    "POST",
                    "/alerts/add/",
                    payload=alert_payload,
                    timeout=5  # 5 second timeout to avoid blocking
                )
                api_call_duration = time.time() - start_time

                if api_response and api_response.status_code == 200:
                    print(
                        f"Alert sent (latency: {ingestion_latency:.2f}s, API: {api_call_duration:.2f}s) for transaction: "
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
        uncertainty_low, uncertainty_high = self._load_uncertainty_zone()

        sanctions_rows = batch_df.filter(col("sanctions_hit") == 1).collect()
        non_sanctions_df = batch_df.filter(col("sanctions_hit") != 1)
        high_risk_rows = non_sanctions_df.filter(col("ml_probability") > 0.85).collect()
        sampled_low_risk_rows = []
        uncertainty_rows = []

        if random_sample_rate > 0:
            low_risk_df = non_sanctions_df.filter(col("ml_probability") < random_sample_max_prob)
            if random_sample_rate < 1:
                low_risk_df = low_risk_df.sample(withReplacement=False, fraction=random_sample_rate)
            sampled_low_risk_rows = low_risk_df.collect()

        uncertainty_rows = non_sanctions_df.filter(
            (col("ml_probability") >= uncertainty_low) & (col("ml_probability") <= uncertainty_high)
        ).collect()

        if not sanctions_rows and not high_risk_rows and not sampled_low_risk_rows and not uncertainty_rows:
            return

        from rag_engine.sar_generator import SARGenerator
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

        sar_gen = SARGenerator() if sanctions_rows or high_risk_rows else None

        max_workers, async_timeout = self._load_alerting_config()
        print(
            f"Processing {len(sanctions_rows)} sanctions hits, "
            f"{len(high_risk_rows)} high-risk alerts, "
            f"{len(sampled_low_risk_rows)} random samples, and "
            f"{len(uncertainty_rows)} uncertainty samples with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {}

            for row in sanctions_rows:
                future = executor.submit(self._send_alert_async, row, sar_gen, "SANCTIONS_HIT", True)
                future_to_row[future] = row

            for row in high_risk_rows:
                future = executor.submit(self._send_alert_async, row, sar_gen, "high_risk", True)
                future_to_row[future] = row

            for row in sampled_low_risk_rows:
                future = executor.submit(self._send_alert_async, row, sar_gen, "random_sample", False)
                future_to_row[future] = row

            for row in uncertainty_rows:
                future = executor.submit(self._send_alert_async, row, sar_gen, "uncertainty_sample", False)
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

        try:
            new_feedback_count = self._count_new_feedback_since_promotion()
            threshold = max(1, self._parse_int_env("RETRAIN_FEEDBACK_THRESHOLD", 100))
            if new_feedback_count < threshold:
                return

            print(f"Triggering model retraining based on {new_feedback_count} new feedback records")

            retrain_response = make_authenticated_request(
                "POST",
                "/retrain-model/",
                timeout=10  # 10 second timeout for retraining trigger
            )

            if retrain_response and retrain_response.status_code == 200:
                print("Model retraining triggered successfully")
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

        # Apply channel-specific and account-type rules
        enriched_with_d17 = apply_d17_rule(enriched)
        enriched_with_d17 = apply_tunicheque_rules(enriched_with_d17)
        enriched_with_d17 = apply_ttn_rules(enriched_with_d17)
        enriched_with_d17 = apply_fcy_rules(enriched_with_d17)
        enriched_with_d17 = apply_device_behavior_rules(enriched_with_d17)
        enriched_with_d17 = self._apply_sanctions_screening(enriched_with_d17)

        # Complex Windowing: Velocity + Multi-Gov
        analytics = enriched_with_d17.groupBy(
            window(col("event_time"), "5 minutes", "1 minute"),
            col("user_id")
        ).agg(
            count("transaction_id").alias("v_count"),
            approx_count_distinct("governorate").alias("g_dist"),
            lit(None).cast(DoubleType()).alias("amount_tnd")  # Placeholder for avg amount
        )

        # Calculate per-user window aggregates.
        # Boolean risk flags from channel rules are max-aggregated so that a single
        # flagged transaction in the window elevates the entire user window's score.
        analytics_with_amount = enriched_with_d17.groupBy(
            window(col("event_time"), "5 minutes", "1 minute"),
            col("user_id")
        ).agg(
            count("transaction_id").alias("v_count"),
            approx_count_distinct("governorate").alias("g_dist"),
            expr("avg(amount_tnd)").alias("avg_amount"),
            expr("max_by(transaction_id, event_time)").alias("transaction_id"),
            expr("max_by(timestamp, event_time)").alias("timestamp"),
            expr("max_by(governorate, event_time)").alias("governorate"),
            expr("max_by(payment_method, event_time)").alias("payment_method"),
            expr("max_by(branch_id, event_time)").alias("branch_id"),
            expr("max_by(sender_account, event_time)").alias("sender_account"),
            expr("max_by(receiver_account, event_time)").alias("receiver_account"),
            expr("max(case when sanctions_hit_flag = true then 1 else 0 end)").alias("sanctions_hit"),
            expr("max(case when pep_connected = true then 1 else 0 end)").alias("pep_connected_flag"),
            expr("sum(case when payment_method = 'Flouci' then 1 else 0 end)").alias("flouci_count"),
            # TuniChèque flags
            expr("max(case when tunicheque_missing_token_flag = true then 1 else 0 end)").alias("tunicheque_missing_token"),
            expr("max(case when tunicheque_expired_lock_flag = true then 1 else 0 end)").alias("tunicheque_expired_lock"),
            # TTN e-invoicing flags
            expr("max(case when ttn_missing_token_flag = true then 1 else 0 end)").alias("ttn_missing_token"),
            expr("max(case when ttn_missing_invoice_id_flag = true then 1 else 0 end)").alias("ttn_missing_invoice_id"),
            # FCY layering flags (Finance Law 2026)
            expr("max(case when fcy_round_amount_flag = true then 1 else 0 end)").alias("fcy_round_amount"),
            expr("max(case when fcy_large_credit_flag = true then 1 else 0 end)").alias("fcy_large_credit"),
            # FCY multi-sender smurfing: count of distinct users sending to FCY accounts in window
            expr("sum(case when account_type = 'FCY' then 1 else 0 end)").alias("fcy_tx_count"),
            # Device fingerprinting / behavioral biometrics flags
            expr("max(case when device_vpn_new_high_amount_flag = true then 1 else 0 end)").alias("device_vpn_new_high_amount"),
            expr("max(case when device_emulator_flag = true then 1 else 0 end)").alias("device_emulator"),
            expr("max(case when device_shared_accounts_flag = true then 1 else 0 end)").alias("device_shared_accounts"),
        )

        # Fetch live rule thresholds from the rules engine.
        # Values are captured at query-plan build time; the rules engine's 30-second
        # TTL cache handles hot-reload on the next Spark driver restart.
        _engine = get_rules_engine()
        _high_value_threshold = _engine.get_high_value_threshold()
        _smurfing = _engine.get_smurfing_params()

        # 3. Weighted Risk Scoring (The Industrial Logic)
        scored = analytics_with_amount.withColumn(
            "velocity_risk",
            when(col("v_count") > 3, lit(1.0)).otherwise(lit(0.0))
        ).withColumn(
            "travel_risk",
            when(col("g_dist") > 1, lit(1.0)).otherwise(lit(0.0))
        ).withColumn(
            # Finance Law 2026: TND 5,000 cash cap repealed. Threshold now sourced
            # from the rules engine (default 15,000 TND) as a general large-tx flag.
            "high_value_risk",
            when(col("avg_amount") > lit(_high_value_threshold), lit(1.0)).otherwise(lit(0.0))
        ).withColumn(
            "d17_risk",
            when((col("avg_amount") > 2000) & (col("flouci_count") > 0), lit(1.0)).otherwise(lit(0.0))
        ).withColumn(
            # Velocity-based smurfing: multiple sub-threshold txs whose window
            # aggregate exceeds the minimum — does not depend on any hard cash cap.
            "smurfing_velocity_risk",
            when(
                (col("v_count") >= lit(_smurfing["min_count"])) &
                (col("avg_amount") < lit(_smurfing["unit_cap"])) &
                (col("v_count") * col("avg_amount") > lit(_smurfing["agg_min"])),
                lit(1.0)
            ).otherwise(lit(0.0))
        ).withColumn(
            # FCY layering: round-amount conversion or large single credit into FCY account
            "fcy_risk",
            when(
                (col("fcy_round_amount") == lit(1)) |
                (col("fcy_large_credit") == lit(1)) |
                (col("fcy_tx_count") >= lit(3)),
                lit(1.0)
            ).otherwise(lit(0.0))
        ).withColumn(
            # TuniChèque: missing QR token = counterfeit or pre-reform cheque (high risk)
            "tunicheque_risk",
            when(
                (col("tunicheque_missing_token") == lit(1)) |
                (col("tunicheque_expired_lock") == lit(1)),
                lit(1.0)
            ).otherwise(lit(0.0))
        ).withColumn(
            # TTN e-invoicing: missing clearance token or invoice ID = non-compliant or fabricated
            "ttn_risk",
            when(
                (col("ttn_missing_token") == lit(1)) |
                (col("ttn_missing_invoice_id") == lit(1)),
                lit(1.0)
            ).otherwise(lit(0.0))
        ).withColumn(
            "pep_risk",
            when(col("pep_connected_flag") == lit(1), lit(1.0)).otherwise(lit(0.0))
        ).withColumn(
            # Device risk: emulator, fresh-device VPN high amount, or shared-device mule pattern
            "device_risk",
            when(
                (col("device_vpn_new_high_amount") == lit(1)) |
                (col("device_emulator") == lit(1)) |
                (col("device_shared_accounts") == lit(1)),
                lit(1.0)
            ).otherwise(lit(0.0))
        ).withColumn(
            "risk_score",
            (col("sanctions_hit") * lit(1.0)) +
            (col("pep_risk") * RISK_WEIGHTS["high_value"]) +
            (col("velocity_risk") * RISK_WEIGHTS["velocity"]) +
            (col("travel_risk") * RISK_WEIGHTS["travel"]) +
            (col("high_value_risk") * RISK_WEIGHTS["high_value"]) +
            (col("d17_risk") * RISK_WEIGHTS["d17_limit"]) +
            (col("smurfing_velocity_risk") * RISK_WEIGHTS["d17_limit"]) +
            # Channel compliance risks use high_value weight (0.2) as they are definitive signals
            (col("tunicheque_risk") * RISK_WEIGHTS["high_value"]) +
            (col("ttn_risk") * RISK_WEIGHTS["high_value"]) +
            # FCY layering: round amount or large single credit into FCY account
            (col("fcy_risk") * RISK_WEIGHTS["high_value"]) +
            # Device/behavioral biometric risks are strong EDD signals.
            (col("device_risk") * RISK_WEIGHTS["high_value"])
        )

        # Prepare features for ML model regardless of model availability to ensure consistent schema
        features_df = scored \
            .withColumn(
                # D17 e-wallet smurfing: amount in the 1400–1500 TND range (just below D17 soft limit).
                # Kept separate from smurfing_velocity_risk which is payment-method agnostic.
                "is_smurfing",
                when(col("avg_amount").between(D17_SOFT_LIMIT - 100, D17_SOFT_LIMIT), 1).otherwise(0)
            ) \
            .withColumn("smurfing_velocity_flag", col("smurfing_velocity_risk").cast("integer")) \
            .withColumn("high_velocity_flag", when(col("v_count") > D17_VELOCITY_CAP, 1).otherwise(0))

        # Apply ML inference if model is available
        if self.ml_model:
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.functions import vector_to_array
            from pyspark.ml import PipelineModel

            if isinstance(self.ml_model, PipelineModel):
                predictions = self.ml_model.transform(features_df)
            else:
                # Create feature vector for ML model
                assembler = VectorAssembler(
                    inputCols=MODEL_FEATURE_COLS,
                    outputCol="features"
                )
                assembled_df = assembler.transform(features_df)
                predictions = self.ml_model.transform(assembled_df)
            final_df = predictions.withColumnRenamed("prediction", "ml_prediction") \
                                  .withColumn("ml_probability", vector_to_array(col("probability")).getItem(1)) \
                                  .drop("probability")
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
