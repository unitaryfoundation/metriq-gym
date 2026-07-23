"""Canonical platform identifiers shared across runtime and export paths."""

AWS_PROVIDER = "aws"
AWS_PROVIDER_ALIASES = frozenset({AWS_PROVIDER, "braket"})


def canonical_provider_name(provider: str) -> str:
    """Return the canonical provider identifier while accepting known aliases."""
    stripped = provider.strip()
    if stripped.lower() in AWS_PROVIDER_ALIASES:
        return AWS_PROVIDER
    return stripped


def canonical_device_name(provider: str, device: str) -> str:
    """Normalize device identifiers whose provider has a canonical representation."""
    stripped = device.strip()
    if canonical_provider_name(provider) != AWS_PROVIDER:
        return stripped

    # Braket device IDs are ARNs. Dataset identifiers use the final two ARN path
    # segments, joined by an underscore (for example, ``iqm_emerald``).
    parts = [part for part in stripped.split("/") if part]
    if len(parts) >= 2:
        stripped = f"{parts[-2]}_{parts[-1]}"
    return stripped.lower()
