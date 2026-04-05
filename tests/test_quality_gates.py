"""Tests for data quality gates."""
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType

from shared.quality_gates import validate_transaction_quality, apply_d17_rule


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder \
        .master("local[1]") \
        .appName("TestQualityGates") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def tx_schema():
    return StructType([
        StructField("transaction_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("amount_tnd", DoubleType(), True),
        StructField("governorate", StringType(), True),
        StructField("payment_method", StringType(), True),
        StructField("branch_id", StringType(), True),
        StructField("fraud_seed", BooleanType(), True),
    ])


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
            "Tunis", "Sfax", "Sousse", "Ariana", "Bizerte", "Gabes", "Kairouan",
            "Manouba", "Ben Arous", "Nabeul", "Zaghouan", "Monastir", "Mahdia",
            "Kasserine", "Sidi Bouzid", "Gafsa", "Tozeur", "Kebili", "Medenine",
            "Tataouine", "Jendouba", "Beja", "Le Kef", "Siliana"
        ]
        data = [(f"TXN_{i}", "2026-01-01T00:00:00Z", "U1", 500.0, gov, "Flouci", "B01", False)
                for i, gov in enumerate(governorates)]
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
