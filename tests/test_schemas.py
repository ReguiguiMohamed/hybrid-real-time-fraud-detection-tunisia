"""Tests for Pydantic schemas and Spark schema conversion."""
import pytest
from shared.schemas import Transaction, pydantic_to_spark_schema, TRANSACTION_SPARK_SCHEMA


class TestTransactionSchema:
    def test_create_valid_transaction(self, sample_transaction_dict):
        tx = Transaction(**sample_transaction_dict)
        assert tx.user_id == "USER_1234"
        assert tx.amount_tnd == 2500.00
        assert tx.governorate == "Tunis"
        assert tx.payment_method == "Flouci"
        assert tx.fraud_seed is False

    def test_auto_generated_fields(self):
        tx = Transaction(
            user_id="USER_001",
            amount_tnd=100.0,
            governorate="Sfax",
            payment_method="eDinar",
            branch_id="Sfax-Agency",
        )
        assert tx.transaction_id is not None
        assert len(tx.transaction_id) > 0
        assert tx.timestamp is not None

    def test_model_dump(self, sample_transaction_dict):
        tx = Transaction(**sample_transaction_dict)
        dumped = tx.model_dump()
        assert isinstance(dumped, dict)
        assert "transaction_id" in dumped
        assert "amount_tnd" in dumped

    def test_fraud_seed_default(self):
        tx = Transaction(
            user_id="USER_001",
            amount_tnd=100.0,
            governorate="Tunis",
            payment_method="eDinar",
            branch_id="Tunis-GNC",
        )
        assert tx.fraud_seed is False


class TestSparkSchemaConversion:
    def test_schema_has_correct_fields(self):
        schema = TRANSACTION_SPARK_SCHEMA
        field_names = [f.name for f in schema.fields]
        expected = ["transaction_id", "timestamp", "user_id", "amount_tnd",
                     "governorate", "payment_method", "branch_id", "fraud_seed"]
        assert field_names == expected

    def test_amount_is_double_type(self):
        from pyspark.sql.types import DoubleType
        schema = TRANSACTION_SPARK_SCHEMA
        amount_field = next(f for f in schema.fields if f.name == "amount_tnd")
        assert isinstance(amount_field.dataType, DoubleType)

    def test_pydantic_to_spark_returns_struct_type(self):
        from pyspark.sql.types import StructType
        result = pydantic_to_spark_schema(Transaction)
        assert isinstance(result, StructType)
