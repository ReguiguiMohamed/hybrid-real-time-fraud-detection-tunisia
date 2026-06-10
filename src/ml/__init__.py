"""Machine-learning components with optional backends loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "DriftDetector": ("ml.train_model", "DriftDetector"),
    "FraudModelTrainer": ("ml.train_model", "FraudModelTrainer"),
    "FraudDetectionPipeline": ("ml.train_pipeline", "FraudDetectionPipeline"),
    "ShadowModelManager": ("ml.shadow_model", "ShadowModelManager"),
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
