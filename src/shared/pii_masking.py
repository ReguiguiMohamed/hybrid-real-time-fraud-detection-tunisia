"""
PII Data Masking & Anonymization Layer for Amastan Fraud Shield Guard
Complies with GDPR and Tunisian Personal Data Protection Law (Law 2004-63 / INPDP)

Provides:
- SHA-256 hashing for user IDs and transaction identifiers
- Amount perturbation for aggregate reporting
- Governorate generalization for external exports
- K-anonymity checking before data release
"""
import hashlib
import hmac
import os
from typing import Optional
from datetime import datetime


# Salt key for HMAC-based hashing (should be loaded from secret vault in production)
_PIISALT = os.getenv("PII_SALT_KEY", "amastan-default-salt-change-in-production")


def hash_pii(value: str, salt: Optional[str] = None) -> str:
    """
    Deterministically hash a PII value for anonymized lookups.
    Uses HMAC-SHA256 for added security.

    Args:
        value: The PII value to hash (e.g., user_id, email, phone)
        salt: Optional salt override. Defaults to env-based salt.

    Returns:
        Hex-encoded SHA-256 HMAC digest.
    """
    if not value:
        return ""
    s = salt or _PIISALT
    return hmac.new(s.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def mask_email(email: str) -> str:
    """
    Mask an email address for display purposes.
    Example: ahmed.benali@example.com -> a***d@example.com
    """
    if not email or "@" not in email:
        return "***"
    local, domain = email.rsplit("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
    return f"{masked_local}@{domain}"


def mask_phone(phone: str) -> str:
    """
    Mask a phone number, keeping only last 4 digits visible.
    Example: +21698765432 -> *******5432
    """
    if not phone:
        return "***"
    clean = phone.strip().replace(" ", "").replace("-", "")
    if len(clean) <= 4:
        return "*" * len(clean)
    return "*" * (len(clean) - 4) + clean[-4:]


def mask_amount(amount: float, precision: int = 2) -> float:
    """
    Round amount to a less precise value for aggregate reporting.
    Example: 1547.23 with precision=2 -> 1500.0
    """
    if amount <= 0:
        return 0.0
    divisor = 10 ** precision
    return round((amount // divisor) * divisor, 0)


def generalize_governorate(governorate: str) -> str:
    """
    Generalize governorate to region level for external reporting.
    Maps 24 governorates to 6 broad regions.
    """
    region_map = {
        "Tunis": "Grand Tunis",
        "Ariana": "Grand Tunis",
        "Ben Arous": "Grand Tunis",
        "Manouba": "Grand Tunis",
        "Nabeul": "Cap Bon",
        "Zaghouan": "Cap Bon",
        "Bizerte": "Nord",
        "Beja": "Nord-Ouest",
        "Jendouba": "Nord-Ouest",
        "Le Kef": "Nord-Ouest",
        "Siliana": "Nord-Ouest",
        "Sousse": "Sahel",
        "Monastir": "Sahel",
        "Mahdia": "Sahel",
        "Sfax": "Sud-Est",
        "Gabes": "Sud-Est",
        "Medenine": "Sud-Est",
        "Tataouine": "Sud-Est",
        "Kairouan": "Centre",
        "Kasserine": "Centre-Ouest",
        "Sidi Bouzid": "Centre-Ouest",
        "Gafsa": "Sud-Ouest",
        "Tozeur": "Sud-Ouest",
        "Kebili": "Sud-Ouest",
    }
    return region_map.get(governorate, "Unknown")


def anonymize_transaction(tx_data: dict) -> dict:
    """
    Create an anonymized copy of a transaction dictionary.
    - Hashes user_id
    - Generalizes governorate
    - Masks amount for external use

    Args:
        tx_data: Original transaction dictionary

    Returns:
        New dictionary with anonymized fields.
    """
    anonymized = tx_data.copy()
    if "user_id" in anonymized:
        anonymized["user_id_hashed"] = hash_pii(anonymized["user_id"])
        anonymized["user_id"] = "***REDACTED***"
    if "governorate" in anonymized:
        anonymized["governorate_region"] = generalize_governorate(anonymized["governorate"])
    if "amount_tnd" in anonymized:
        anonymized["amount_tnd_masked"] = mask_amount(anonymized["amount_tnd"])
    if "transaction_id" in anonymized:
        anonymized["transaction_id_hashed"] = hash_pii(anonymized["transaction_id"])
    return anonymized


def check_k_anonymity(data: list, quasi_identifiers: list, k: int = 5) -> bool:
    """
    Check if a dataset satisfies k-anonymity.
    Every combination of quasi-identifier values must appear in at least k records.

    Args:
        data: List of dictionaries
        quasi_identifiers: Columns that could be used to re-identify (e.g., ['governorate', 'payment_method'])
        k: Minimum group size (default 5)

    Returns:
        True if k-anonymity is satisfied.
    """
    from collections import Counter

    # Extract quasi-identifier tuples
    qi_tuples = []
    for row in data:
        qi_tuples.append(tuple(row.get(qi, None) for qi in quasi_identifiers))

    # Count occurrences
    counts = Counter(qi_tuples)

    # Check minimum group size
    return all(count >= k for count in counts.values())


def generate_data_retention_policy(data_type: str) -> dict:
    """
    Returns data retention policy based on data type and Tunisian law requirements.

    Args:
        data_type: Type of data (e.g., 'transaction', 'alert', 'feedback', 'sar_report')

    Returns:
        Dictionary with retention rules.
    """
    policies = {
        "transaction": {
            "retention_days": 365 * 5,  # 5 years per BCT regulations
            "anonymize_after_days": 365 * 2,
            "legal_basis": "BCT banking record retention requirement",
        },
        "alert": {
            "retention_days": 365 * 5,
            "anonymize_after_days": 365 * 2,
            "legal_basis": "CTAF AML/CFT compliance",
        },
        "feedback": {
            "retention_days": 365 * 3,
            "anonymize_after_days": 365,
            "legal_basis": "Model training necessity",
        },
        "sar_report": {
            "retention_days": 365 * 10,  # 10 years for SARs
            "anonymize_after_days": None,  # SARs must not be anonymized
            "legal_basis": "CTAF SAR retention requirement",
        },
        "dlq": {
            "retention_days": 90,
            "anonymize_after_days": 30,
            "legal_basis": "Operational error log retention",
        },
    }
    return policies.get(data_type, {"retention_days": 365, "anonymize_after_days": 180, "legal_basis": "Default policy"})
