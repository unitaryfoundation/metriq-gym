"""IBM Sampler provider integration.

This package provides a session-aware IBM device that supports
twirling options and parameterized circuit execution via SamplerV2.
It wraps qBraid's QiskitBackend with an overridden submit() that
gives users the options to run inside a session context, and use
parameterized or twirled circuits via the SamplerV2 interface.
"""

__all__ = [
    "provider",
    "device",
]
