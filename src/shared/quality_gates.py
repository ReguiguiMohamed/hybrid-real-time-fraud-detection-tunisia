"""
Quality gates for fraud detection pipeline using Great Expectations principles
"""
from pyspark.sql.functions import col, when, lit, isnan, isnull

def validate_transaction_quality(df):
    """
    Apply data quality checks to transaction DataFrame
    This function adds quality validation columns but doesn't perform counts on streaming data
    """
    # Add quality validation columns to the streaming dataframe
    df_validated = df.withColumn(
        "negative_amount_flag",
        when(col("amount_tnd") < 0, lit(True)).otherwise(lit(False))
    ).withColumn(
        "invalid_governorate_flag",
        when(~col("governorate").isin(["Tunis", "Sfax", "Sousse", "Ariana", "Bizerte", "Gabes", "Kairouan"]), lit(True)).otherwise(lit(False))
    ).withColumn(
        "null_id_flag",
        when(col("transaction_id").isNull(), lit(True)).otherwise(lit(False))
    )

    # Filter out records that fail quality checks
    df_filtered = df_validated.filter(
        (~col("negative_amount_flag")) &
        (~col("invalid_governorate_flag")) &
        (~col("null_id_flag"))
    )

    return df_filtered

def apply_d17_rule(df):
    """
    Apply D17-specific rule: If payment_method is 'Flouci' and amount_tnd > 2000,
    boost risk score by 0.2
    """
    df_with_d17_flag = df.withColumn(
        "d17_risk_boost",
        when((col("payment_method") == "Flouci") & (col("amount_tnd") > 2000), lit(0.2))
        .otherwise(lit(0.0))
    )

    return df_with_d17_flag