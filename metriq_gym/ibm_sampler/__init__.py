"""IBM Sampler provider integration.

This package provides a session-aware IBM device that supports
twirling options and parameterized circuit execution via SamplerV2.
It wraps qBraid's QiskitBackend with an overridden submit() that
always runs inside a Session.
"""

__all__ = [
    "provider",
    "device",
]
