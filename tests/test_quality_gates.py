"""Tests for data quality gates."""

import os
import sys

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType, DoubleType, StringType, StructField, StructType

from shared.quality_gates import apply_d17_rule, apply_device_behavior_rules, validate_transaction_quality

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SPARK_TESTS") != "1",
    reason="Spark integration tests are opt-in; set RUN_SPARK_TESTS=1 to run them.",
)


@pytest.fixture(scope="module")
def spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[1]")
        .appName("TestQualityGates")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def tx_schema():
    return StructType(
        [
            StructField("transaction_id", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("amount_tnd", DoubleType(), True),
            StructField("governorate", StringType(), True),
            StructField("payment_method", StringType(), True),
            StructField("branch_id", StringType(), True),
            StructField("fraud_seed", BooleanType(), True),
        ]
    )


class TestValidateTransactionQuality:
    def test_valid_records_pass(self, spark, tx_schema):
        data = [("TXN1", "2026-01-01T00:00:00Z", "U1", 500.0, "Tunis", "Flouci", "B01", False)]
        df = spark.createDataFrame(data, schema=tx_schema)
        result = validate_transaction_quality(df)
        assert result.count() == 1

    def test_negative_amount_filtered(self, spark, tx_schema):
        data = [("TXN1", "2026-01-01T00:00:00Z", "U1", -100.0, "Tunis", "Flouci", "B01", False)]
        df = spark.createDataFrame(data, schema=tx_schema)
        result = validate_transaction_quality(df)
        assert result.count() == 0

    def test_invalid_governorate_filtered(self, spark, tx_schema):
        data = [("TXN1", "2026-01-01T00:00:00Z", "U1", 500.0, "InvalidCity", "Flouci", "B01", False)]
        df = spark.createDataFrame(data, schema=tx_schema)
        result = validate_transaction_quality(df)
        assert result.count() == 0

    def test_null_transaction_id_filtered(self, spark, tx_schema):
        data = [(None, "2026-01-01T00:00:00Z", "U1", 500.0, "Tunis", "Flouci", "B01", False)]
        df = spark.createDataFrame(data, schema=tx_schema)
        result = validate_transaction_quality(df)
        assert result.count() == 0

    def test_all_24_governorates_accepted(self, spark, tx_schema):
        governorates = [
            "Tunis",
            "Sfax",
            "Sousse",
            "Ariana",
            "Bizerte",
            "Gabes",
            "Kairouan",
            "Manouba",
            "Ben Arous",
            "Nabeul",
            "Zaghouan",
            "Monastir",
            "Mahdia",
            "Kasserine",
            "Sidi Bouzid",
            "Gafsa",
            "Tozeur",
            "Kebili",
            "Medenine",
            "Tataouine",
            "Jendouba",
            "Beja",
            "Le Kef",
            "Siliana",
        ]
        data = [
            (f"TXN_{i}", "2026-01-01T00:00:00Z", "U1", 500.0, gov, "Flouci", "B01", False)
            for i, gov in enumerate(governorates)
        ]
        df = spark.createDataFrame(data, schema=tx_schema)
        result = validate_transaction_quality(df)
        assert result.count() == len(governorates)


class TestApplyD17Rule:
    def test_flouci_high_amount_gets_boost(self, spark, tx_schema):
        data = [("TXN1", "2026-01-01T00:00:00Z", "U1", 3000.0, "Tunis", "Flouci", "B01", False)]
        df = spark.createDataFrame(data, schema=tx_schema)
        result = apply_d17_rule(df)
        boost = result.collect()[0]["d17_risk_boost"]
        assert boost == 0.2

    def test_non_flouci_no_boost(self, spark, tx_schema):
        data = [("TXN1", "2026-01-01T00:00:00Z", "U1", 3000.0, "Tunis", "eDinar", "B01", False)]
        df = spark.createDataFrame(data, schema=tx_schema)
        result = apply_d17_rule(df)
        boost = result.collect()[0]["d17_risk_boost"]
        assert boost == 0.0

    def test_flouci_low_amount_no_boost(self, spark, tx_schema):
        data = [("TXN1", "2026-01-01T00:00:00Z", "U1", 500.0, "Tunis", "Flouci", "B01", False)]
        df = spark.createDataFrame(data, schema=tx_schema)
        result = apply_d17_rule(df)
        boost = result.collect()[0]["d17_risk_boost"]
        assert boost == 0.0


@pytest.fixture
def device_schema():
    return StructType(
        [
            StructField("transaction_id", StringType(), True),
            StructField("amount_tnd", DoubleType(), True),
            StructField("device_id", StringType(), True),
            StructField("vpn_detected", BooleanType(), True),
            StructField("emulator_detected", BooleanType(), True),
            StructField("device_age_days", DoubleType(), True),
            StructField("device_account_count_7d", DoubleType(), True),
        ]
    )


class TestApplyDeviceBehaviorRules:
    def test_vpn_new_device_high_amount_flag(self, spark, device_schema):
        data = [("TXN_DEVICE_1", 1500.0, "DEV1", True, False, 0.0, 1.0)]
        df = spark.createDataFrame(data, schema=device_schema)

        result = apply_device_behavior_rules(df).collect()[0]

        assert result["device_vpn_new_high_amount_flag"] is True
        assert result["device_emulator_flag"] is False
        assert result["device_shared_accounts_flag"] is False

    def test_emulator_flag(self, spark, device_schema):
        data = [("TXN_DEVICE_2", 200.0, "DEV2", False, True, 30.0, 1.0)]
        df = spark.createDataFrame(data, schema=device_schema)

        result = apply_device_behavior_rules(df).collect()[0]

        assert result["device_emulator_flag"] is True

    def test_shared_device_account_velocity_flag(self, spark, device_schema):
        data = [("TXN_DEVICE_3", 200.0, "DEV3", False, False, 10.0, 4.0)]
        df = spark.createDataFrame(data, schema=device_schema)

        result = apply_device_behavior_rules(df).collect()[0]

        assert result["device_shared_accounts_flag"] is True

    def test_old_non_vpn_device_no_flag(self, spark, device_schema):
        data = [("TXN_DEVICE_4", 1500.0, "DEV4", False, False, 30.0, 1.0)]
        df = spark.createDataFrame(data, schema=device_schema)

        result = apply_device_behavior_rules(df).collect()[0]

        assert result["device_vpn_new_high_amount_flag"] is False
        assert result["device_emulator_flag"] is False
        assert result["device_shared_accounts_flag"] is False
