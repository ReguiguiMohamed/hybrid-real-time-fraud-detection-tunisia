from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Any

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
