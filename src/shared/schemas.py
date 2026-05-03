from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, get_args, get_origin

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

    # ── Device / behavioral biometrics signals (nullable — absent from API-originated transactions) ──
    device_id: Optional[str] = Field(None, description="Hashed device fingerprint.")
    device_os: Optional[str] = Field(None, description="Client device operating system, when supplied by the channel.")
    device_model: Optional[str] = Field(None, description="Client device model, when supplied by the channel.")
    app_version: Optional[str] = Field(None, description="Mobile or web application version used for the transaction.")
    session_typing_cadence_ms: Optional[float] = Field(
        None,
        description="Median inter-keystroke delay in milliseconds for the authenticated session.",
    )
    session_copy_paste_ratio: Optional[float] = Field(
        None,
        description="Ratio of pasted input events to total text-entry events in the authenticated session.",
    )
    network_type: Optional[str] = Field(None, description="Observed network type: 4G, 5G, WIFI, VPN, TOR, or similar.")
    vpn_detected: Optional[bool] = Field(None, description="True if transaction originated via VPN.")
    emulator_detected: Optional[bool] = Field(None, description="True if mobile emulator detected.")
    device_age_days: Optional[float] = Field(None, description="Days since this device fingerprint was first observed.")
    device_account_count_7d: Optional[float] = Field(
        None,
        description="Distinct accounts seen from this device fingerprint over the last 7 days.",
    )

    # Sanctions / PEP screening inputs.
    sender_account: Optional[str] = Field(None, description="Originating account identifier for sanctions screening.")
    receiver_account: Optional[str] = Field(None, description="Beneficiary account identifier for sanctions screening.")
    pep_connected: Optional[bool] = Field(None, description="True when account enrichment marks the party as PEP-connected.")


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

    model_fields = getattr(model_class, "model_fields", {})

    def unwrap_optional(field_type):
        origin = get_origin(field_type)
        if origin in (Union, getattr(__import__("types"), "UnionType", None)):
            args = [arg for arg in get_args(field_type) if arg is not type(None)]
            if len(args) == 1:
                return args[0], True
        return field_type, False

    # Get model fields and their types
    fields = []
    for field_name, field_info in model_class.__annotations__.items():
        field = model_fields.get(field_name)
        field_type = field.annotation if field else field_info
        unwrapped_type, is_optional = unwrap_optional(field_type)

        # Map the type to Spark type
        spark_type = StringType()  # Default to StringType
        if unwrapped_type in type_mapping:
            spark_type = type_mapping[unwrapped_type]
        else:
            # For other types like UUID, datetime, etc., use StringType
            spark_type = StringType()

        # Determine if nullable based on whether it has a default value
        if not is_optional and field is not None:
            is_optional = not field.is_required()

        fields.append(StructField(field_name, spark_type, is_optional))

    return StructType(fields)


# Pre-defined schema for Transaction model
TRANSACTION_SPARK_SCHEMA = pydantic_to_spark_schema(Transaction)
