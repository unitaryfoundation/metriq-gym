"""Registry for QEM technique implementations."""

from typing import Any

from metriq_gym.qem.pipeline import MitigationPipeline
from metriq_gym.qem.technique import MitigationTechnique

TECHNIQUE_REGISTRY: dict[str, type[MitigationTechnique]] = {}


def register_technique(name: str):
    """Decorator to register a mitigation technique by name.

    Usage:
        @register_technique("zne")
        class ZNETechnique(MitigationTechnique):
            ...
    """

    def decorator(cls: type[MitigationTechnique]) -> type[MitigationTechnique]:
        TECHNIQUE_REGISTRY[name] = cls
        return cls

    return decorator


def build_pipeline(config_list: list[dict[str, Any]]) -> MitigationPipeline:
    """Build a MitigationPipeline from a list of technique configs.

    Each config dict must have a "technique" key naming the registered technique.
    The remaining keys are passed as technique-specific configuration.

    Args:
        config_list: List of dicts, e.g. [{"technique": "zne", "scale_factors": [1,3,5]}].

    Returns:
        A MitigationPipeline composing the requested techniques in order.

    Raises:
        ValueError: If a technique name is not registered.
    """
    # Ensure technique implementations are registered
    _ensure_techniques_loaded()

    techniques: list[MitigationTechnique] = []
    for config in config_list:
        name = config.get("technique")
        if name is None:
            raise ValueError(f"Mitigation config missing 'technique' key: {config}")
        if name not in TECHNIQUE_REGISTRY:
            available = sorted(TECHNIQUE_REGISTRY.keys())
            raise ValueError(
                f"Unknown mitigation technique: '{name}'. Available: {available}"
            )
        technique_cls = TECHNIQUE_REGISTRY[name]
        techniques.append(technique_cls(config))
    return MitigationPipeline(techniques)


def _ensure_techniques_loaded():
    """Import technique modules to trigger @register_technique decorators."""
    import metriq_gym.qem.techniques  # noqa: F401
