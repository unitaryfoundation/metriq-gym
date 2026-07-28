"""Canonical provider aliases and dataset-facing device identifiers."""

AWS_PROVIDER = "aws"
AWS_PROVIDER_ALIASES = frozenset({AWS_PROVIDER, "braket"})


def canonical_provider_name(provider: str) -> str:
    """Canonicalize known provider aliases while preserving all other identifiers."""
    stripped = provider.strip()
    if stripped.lower() in AWS_PROVIDER_ALIASES:
        return AWS_PROVIDER
    return stripped


def canonical_device_name(provider: str, device: str) -> str:
    """Return a stable dataset key without changing the runtime device address."""
    stripped = device.strip()
    if canonical_provider_name(provider) != AWS_PROVIDER:
        return stripped

    # Braket device IDs are ARNs. Dataset identity intentionally ignores the ARN
    # region and uses the final two path segments (for example, ``iqm_emerald``).
    # The full, case-sensitive runtime ARN remains on MetriqGymJob.device_name.
    parts = [part for part in stripped.split("/") if part]
    if len(parts) >= 2:
        stripped = f"{parts[-2]}_{parts[-1]}"
    return stripped.lower()
