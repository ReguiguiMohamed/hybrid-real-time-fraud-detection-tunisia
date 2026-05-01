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
    # Core fields that must always be present (subset check, not exact equality,
    # so adding new Optional fields to Transaction doesn't break this test).
    CORE_FIELDS = [
        "transaction_id", "timestamp", "user_id", "amount_tnd",
        "governorate", "payment_method", "branch_id", "fraud_seed",
    ]
    # New fields added for Finance Law 2026 / TuniChèque / TTN / FCY compliance
    COMPLIANCE_FIELDS = [
        "tunicheque_token", "tunicheque_provision_locked", "tunicheque_clearing_deadline",
        "ttn_clearance_token", "ttn_invoice_id",
        "account_type", "fcy_currency",
        "device_id", "vpn_detected", "emulator_detected",
    ]

    def test_schema_contains_all_core_fields(self):
        field_names = [f.name for f in TRANSACTION_SPARK_SCHEMA.fields]
        for name in self.CORE_FIELDS:
            assert name in field_names, f"Core field '{name}' missing from Spark schema"

    def test_schema_contains_all_compliance_fields(self):
        field_names = [f.name for f in TRANSACTION_SPARK_SCHEMA.fields]
        for name in self.COMPLIANCE_FIELDS:
            assert name in field_names, f"Compliance field '{name}' missing from Spark schema"

    def test_all_compliance_fields_are_nullable(self):
        """Optional fields in Transaction must be nullable in the Spark schema."""
        fields_by_name = {f.name: f for f in TRANSACTION_SPARK_SCHEMA.fields}
        for name in self.COMPLIANCE_FIELDS:
            assert fields_by_name[name].nullable, f"Compliance field '{name}' must be nullable"

    def test_amount_is_double_type(self):
        from pyspark.sql.types import DoubleType
        amount_field = next(f for f in TRANSACTION_SPARK_SCHEMA.fields if f.name == "amount_tnd")
        assert isinstance(amount_field.dataType, DoubleType)

    def test_pydantic_to_spark_returns_struct_type(self):
        from pyspark.sql.types import StructType
        result = pydantic_to_spark_schema(Transaction)
        assert isinstance(result, StructType)


class TestTransactionOptionalFields:
    """Ensure new optional fields default to None and don't break existing construction."""

    def test_optional_fields_default_none(self):
        tx = Transaction(
            user_id="U1", amount_tnd=100.0,
            governorate="Tunis", payment_method="Flouci", branch_id="B1",
        )
        assert tx.tunicheque_token is None
        assert tx.ttn_clearance_token is None
        assert tx.account_type is None
        assert tx.fcy_currency is None
        assert tx.device_id is None
        assert tx.vpn_detected is None

    def test_tunicheque_fields_accepted(self):
        tx = Transaction(
            user_id="U1", amount_tnd=5000.0,
            governorate="Tunis", payment_method="TUNICHEQUE", branch_id="B1",
            tunicheque_token="QR_ABC123",
            tunicheque_provision_locked=True,
            tunicheque_clearing_deadline="2026-05-09",
        )
        assert tx.tunicheque_token == "QR_ABC123"
        assert tx.tunicheque_provision_locked is True

    def test_ttn_fields_accepted(self):
        tx = Transaction(
            user_id="U1", amount_tnd=2000.0,
            governorate="Sfax", payment_method="TTN_EINVOICE", branch_id="B2",
            ttn_clearance_token="TTN_XYZ789",
            ttn_invoice_id="INV-2026-00042",
        )
        assert tx.ttn_clearance_token == "TTN_XYZ789"
        assert tx.ttn_invoice_id == "INV-2026-00042"

    def test_fcy_fields_accepted(self):
        tx = Transaction(
            user_id="U1", amount_tnd=10000.0,
            governorate="Tunis", payment_method="Virement", branch_id="B1",
            account_type="FCY", fcy_currency="EUR",
        )
        assert tx.account_type == "FCY"
        assert tx.fcy_currency == "EUR"
