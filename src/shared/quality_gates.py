"""
Quality gates for fraud detection pipeline using Great Expectations principles.
Includes channel-specific rules for TuniChèque (Feb 2025) and TTN e-invoicing (Jan 2026).
"""
import os

from pyspark.sql.functions import col, when, lit, isnan, isnull, lower, datediff, to_date, current_date

DEFAULT_FCY_LARGE_CREDIT_TND = 5000.0


def _float_env(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

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
        when(~col("governorate").isin([
            "Tunis", "Sfax", "Sousse", "Ariana", "Bizerte", "Gabes", "Kairouan",
            "Manouba", "Ben Arous", "Nabeul", "Zaghouan", "Monastir", "Mahdia",
            "Kasserine", "Sidi Bouzid", "Gafsa", "Tozeur", "Kebili", "Medenine",
            "Tataouine", "Jendouba", "Beja", "Le Kef", "Siliana"
        ]), lit(True)).otherwise(lit(False))
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

def apply_tunicheque_rules(df):
    """
    TuniChèque fraud rules (Law n°2024-41, platform live Feb 2, 2025).

    Rule 1 — Missing token: A TUNICHEQUE transaction without a tunicheque_token is
    either counterfeit or pre-reform, both are high-severity signals.

    Rule 2 — Provision-lock abuse: A legitimate provision lock followed by rapid
    depletion. Detected here as: provision_locked=True but clearing_deadline already
    passed (issuer failed to present, possible float abuse).
    """
    df = df.withColumn(
        "tunicheque_missing_token_flag",
        when(
            (lower(col("payment_method")) == lit("tunicheque")) &
            (col("tunicheque_token").isNull() | (col("tunicheque_token") == lit(""))),
            lit(True),
        ).otherwise(lit(False)),
    )

    df = df.withColumn(
        "tunicheque_expired_lock_flag",
        when(
            (lower(col("payment_method")) == lit("tunicheque")) &
            col("tunicheque_provision_locked").isNotNull() &
            col("tunicheque_provision_locked") &
            col("tunicheque_clearing_deadline").isNotNull() &
            (datediff(current_date(), to_date(col("tunicheque_clearing_deadline"))) > lit(0)),
            lit(True),
        ).otherwise(lit(False)),
    )

    return df


def apply_ttn_rules(df):
    """
    TTN / El Fatoora e-invoicing fraud rules (Finance Law 2026, effective Jan 1, 2026).

    Rule 1 — Missing clearance token: B2B service transactions must carry a TTN token.
    Absence on a TTN_EINVOICE transaction is either non-compliance or fabrication.

    Rule 2 — Duplicate invoice ID: Same ttn_invoice_id appearing more than once
    indicates replay / double-submission fraud.
    (Note: deduplication requires stateful tracking; this gate flags null/empty tokens
    only — dedup is handled in the stateful streaming layer.)
    """
    df = df.withColumn(
        "ttn_missing_token_flag",
        when(
            (lower(col("payment_method")) == lit("ttn_einvoice")) &
            (col("ttn_clearance_token").isNull() | (col("ttn_clearance_token") == lit(""))),
            lit(True),
        ).otherwise(lit(False)),
    )

    df = df.withColumn(
        "ttn_missing_invoice_id_flag",
        when(
            (lower(col("payment_method")) == lit("ttn_einvoice")) &
            (col("ttn_invoice_id").isNull() | (col("ttn_invoice_id") == lit(""))),
            lit(True),
        ).otherwise(lit(False)),
    )

    return df


def apply_fcy_rules(df):
    """
    Foreign Currency Account (FCY) layering rules.
    Enabled by Finance Law 2026: Tunisian residents may now open FCY accounts.
    BCT implementation circulars are still being drafted — rules are intentionally
    conservative and must be reconfigured once circulars define hard caps.

    Rule 1 — Round-amount TND→FCY conversion: classic layering signal.
    Rule 2 — Multi-sender FCY credit: funds from >= 3 distinct senders arriving at
              an FCY account in a 5-min window (smurfing into FCY).
    Rule 3 — New FCY account + large immediate credit: account flagged as FCY type
              with amount above FCY_LARGE_CREDIT_THRESHOLD_TND in a single transaction.
    """
    # Rule 1: Round-number TND→FCY conversion (amount divisible by 1000 exactly)
    df = df.withColumn(
        "fcy_round_amount_flag",
        when(
            (col("account_type") == lit("FCY")) &
            col("fcy_currency").isNotNull() &
            ((col("amount_tnd") % lit(1000.0)) == lit(0.0)) &
            (col("amount_tnd") >= lit(1000.0)),
            lit(True),
        ).otherwise(lit(False)),
    )

    fcy_large_credit_threshold = _float_env(
        "FCY_LARGE_CREDIT_THRESHOLD_TND",
        DEFAULT_FCY_LARGE_CREDIT_TND,
    )

    # Rule 3: Large single credit into FCY account; threshold is configurable
    # because BCT implementation circulars may later define a hard cap.
    # (Rule 2 — multi-sender — requires stateful cross-user aggregation; handled
    #  in the windowed agg layer in consumer.py via approx_count_distinct on user_id
    #  grouped by branch_id when account_type == FCY)
    df = df.withColumn(
        "fcy_large_credit_flag",
        when(
            (col("account_type") == lit("FCY")) &
            (col("amount_tnd") > lit(fcy_large_credit_threshold)),
            lit(True),
        ).otherwise(lit(False)),
    )

    return df


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
