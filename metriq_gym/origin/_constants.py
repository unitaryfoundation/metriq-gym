"""Shared constants for the Origin provider implementation."""

SIMULATOR_BACKENDS = {
    "full_amplitude",
    "partial_amplitude",
    "single_amplitude",
}

SIMULATOR_MAX_QUBITS = {
    "full_amplitude": 35,
    "partial_amplitude": 68,
    "single_amplitude": 200,
}

API_KEY_ENV = "ORIGIN_API_KEY"

# Common alias exposed by Origin marketing materials / notebooks
PRIMARY_BACKEND_ALIAS = "origin_wukong"
PRIMARY_BACKEND_ID = "WK_C102_400"
