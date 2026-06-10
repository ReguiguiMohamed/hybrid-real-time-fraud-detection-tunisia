"""Shared utilities exposed lazily to avoid importing optional runtimes."""

from importlib import import_module

_EXPORTS = {
    "CBDC_PILOT_GOVERNORATES": ("shared.risk_config", "CBDC_PILOT_GOVERNORATES"),
    "DedupCache": ("shared.idempotency", "DedupCache"),
    "RISK_WEIGHTS": ("shared.risk_config", "RISK_WEIGHTS"),
    "TRANSACTION_SPARK_SCHEMA": ("shared.schemas", "TRANSACTION_SPARK_SCHEMA"),
    "Transaction": ("shared.schemas", "Transaction"),
    "anonymize_transaction": ("shared.pii_masking", "anonymize_transaction"),
    "check_k_anonymity": ("shared.pii_masking", "check_k_anonymity"),
    "get_api_headers": ("shared.utils", "get_api_headers"),
    "get_api_url": ("shared.utils", "get_api_url"),
    "get_dedup_cache": ("shared.idempotency", "get_dedup_cache"),
    "get_kafka_credentials": ("shared.vault_client", "get_kafka_credentials"),
    "get_rules_engine": ("shared.rules_engine", "get_rules_engine"),
    "get_secret": ("shared.vault_client", "get_secret"),
    "get_vault_client": ("shared.vault_client", "get_vault_client"),
    "hash_pii": ("shared.pii_masking", "hash_pii"),
    "initialize_tracing": ("shared.tracing", "initialize_tracing"),
    "make_authenticated_request": ("shared.utils", "make_authenticated_request"),
    "mask_amount": ("shared.pii_masking", "mask_amount"),
    "mask_email": ("shared.pii_masking", "mask_email"),
    "mask_phone": ("shared.pii_masking", "mask_phone"),
    "start_span": ("shared.tracing", "start_span"),
    "tracer": ("shared.tracing", "tracer"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
