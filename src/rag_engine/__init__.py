"""RAG components with optional vector-store dependencies loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "CircuitBreaker": ("rag_engine.circuit_breaker", "CircuitBreaker"),
    "CircuitBreakerConfig": ("rag_engine.circuit_breaker", "CircuitBreakerConfig"),
    "CircuitBreakerError": ("rag_engine.circuit_breaker", "CircuitBreakerError"),
    "CircuitState": ("rag_engine.circuit_breaker", "CircuitState"),
    "CTAFVectorStore": ("rag_engine.vector_store", "CTAFVectorStore"),
    "SARGenerator": ("rag_engine.sar_generator", "SARGenerator"),
    "SARRegulatoryViolation": ("rag_engine.sar_validator", "SARRegulatoryViolation"),
    "SARReport": ("rag_engine.sar_validator", "SARReport"),
    "SARRiskFactor": ("rag_engine.sar_validator", "SARRiskFactor"),
    "SARUrgencyAssessment": ("rag_engine.sar_validator", "SARUrgencyAssessment"),
    "format_sar_report": ("rag_engine.sar_validator", "format_sar_report"),
    "generate_deterministic_fallback": ("rag_engine.sar_validator", "generate_deterministic_fallback"),
    "get_rag_circuit": ("rag_engine.circuit_breaker", "get_rag_circuit"),
    "rag_circuit": ("rag_engine.circuit_breaker", "rag_circuit"),
    "validate_sar_output": ("rag_engine.sar_validator", "validate_sar_output"),
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
