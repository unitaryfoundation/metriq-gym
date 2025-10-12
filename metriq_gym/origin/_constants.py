"""Shared constants for the Origin provider implementation."""

SIMULATOR_BACKENDS = {
    "full_amplitude",
    "partial_amplitude",
    "single_amplitude",
}

BACKEND_ALIASES = {
    # Common alias exposed by Origin marketing materials / notebooks
    "origin_wukong": "WK_C102_400",
}

ALIAS_TO_DISPLAY = {value: key for key, value in BACKEND_ALIASES.items()}

API_KEY_ENV = "ORIGIN_API_KEY"
