from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType


class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: str
    amount_tnd: float
    governorate: str
    payment_method: str
    branch_id: str
    fraud_seed: bool = False

    # ── TuniChèque fields (Law n°2024-41, effective Feb 2, 2025) ──────────────
    # Present only when payment_method == "TUNICHEQUE". Null for all other channels.
    tunicheque_token: Optional[str] = Field(
        None,
        description="QR verification token issued by the TuniChèque platform. "
                    "Absence on a cheque transaction is itself a fraud signal.",
    )
    tunicheque_provision_locked: Optional[bool] = Field(
        None,
        description="True if the cheque amount has been reserved on the issuer's account "
                    "by TuniChèque. False or null on non-cheque transactions.",
    )
    tunicheque_clearing_deadline: Optional[str] = Field(
        None,
        description="ISO date by which the cheque must be presented for clearing "
                    "(maximum 8 business days after the provision-lock date).",
    )

    # ── TTN / El Fatoora e-invoicing fields (Finance Law 2026, effective Jan 1 2026) ──
    # Present only when payment_method == "TTN_EINVOICE". Null for all other channels.
    ttn_clearance_token: Optional[str] = Field(
        None,
        description="Real-time clearance token from the TTN El Fatoora platform. "
                    "All B2B VAT-service transactions must carry this token from Jan 2026.",
    )
    ttn_invoice_id: Optional[str] = Field(
        None,
        description="Unique invoice identifier as registered on the TTN platform.",
    )

    # ── Foreign Currency Account fields (Finance Law 2026, BCT circulars pending) ──
    account_type: Optional[str] = Field(
        None,
        description="Account type: TND | FCY | MIXED. "
                    "FCY accounts newly permitted for Tunisian residents under Finance Law 2026.",
    )
    fcy_currency: Optional[str] = Field(
        None,
        description="ISO 4217 currency code for FCY accounts (e.g., EUR, USD). "
                    "Null for TND accounts.",
    )

    # ── Device / channel signals (nullable — absent from API-originated transactions) ──
    device_id: Optional[str] = Field(None, description="Hashed device fingerprint.")
    vpn_detected: Optional[bool] = Field(None, description="True if transaction originated via VPN.")
    emulator_detected: Optional[bool] = Field(None, description="True if mobile emulator detected.")


def pydantic_to_spark_schema(model_class) -> StructType:
    """
    Convert a Pydantic model to a Spark StructType schema.
    This ensures Single Source of Truth and prevents schema duplication issues.
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType

    # Mapping from Python types to Spark types
    type_mapping = {
        str: StringType(),
        float: DoubleType(),
        bool: BooleanType(),
        int: StringType()  # Using StringType for flexibility with UUIDs and timestamps
    }

    # Get model fields and their types
    fields = []
    for field_name, field_info in model_class.__annotations__.items():
        # Get the default value to determine the type if possible
        if field_name in model_class.__fields__:
            field_type = model_class.__fields__[field_name].annotation
        else:
            field_type = field_info

        # Map the type to Spark type
        spark_type = StringType()  # Default to StringType
        if hasattr(field_type, '__origin__'):  # Handle Optional, Union, etc.
            # For complex types, default to StringType
            spark_type = StringType()
        elif field_type in type_mapping:
            spark_type = type_mapping[field_type]
        else:
            # For other types like UUID, datetime, etc., use StringType
            spark_type = StringType()

        # Determine if nullable based on whether it has a default value
        is_optional = hasattr(field_info, '__origin__') and field_info.__origin__ is type(None)
        if not is_optional and field_name in model_class.__fields__:
            field_default = model_class.__fields__[field_name].default
            is_optional = field_default != ...  # ... means required in Pydantic

        fields.append(StructField(field_name, spark_type, True))

    return StructType(fields)


# Pre-defined schema for Transaction model
TRANSACTION_SPARK_SCHEMA = pydantic_to_spark_schema(Transaction)
