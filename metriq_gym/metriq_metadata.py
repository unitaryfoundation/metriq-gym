"""Mappings for "friendly-name" platforms, methods, tasks, etc. that correspond to the keyed IDs on Metriq."""

# Platform name to Metriq ID mappings:
platforms = {
    # IBM
    "ibm_belem": 26,
    "ibm_bogota": 2,
    "ibm_brisbane": 188,
    "ibm_brussels": 213,
    "ibm_casablanca": 21,
    "ibm_fez": 212,
    "ibm_guadalupe": 19,
    "ibm_hanoi": 13,
    "ibm_jakarta": 24,
    "ibm_kyiv": 167,
    "ibm_kyoto": 194,
    "ibm_lagos": 20,
    "ibm_lima": 27,
    "ibm_manila": 23,
    "ibm_melbourne": 76,
    "ibm_montreal": 11,
    "ibm_nairobi": 119,
    "ibm_nazca": 184,
    "ibm_oslo": 120,
    "ibm_perth": 22,
    "ibm_prague": 56,
    "ibm_quito": 25,
    "ibm_sherbrooke": 211,
    "ibm_tokyo": 66,
    "ibm_torino": 209,
    "ibm_toronto": 3,
    "qasm_simulator": 153,
    # Quantinuum
    "H1": 80,
    "H2": 161,
    # Rigetti
    "aspen_11": 8,
    "aspen_9": 151,
    # IonQ
    "aria": 96,
    "forte": 182,
    "harmony": 137,
}


# Method name to Metriq ID mappings:
methods = {
    # IBM
    "ibm_belem": 78,
    "ibm_bogota": 13,
    "ibm_brisbane": 362,
    "ibm_brussels": 412,
    "ibm_casablanca": 71,
    "ibm_fez": 411,
    "ibm_guadalupe": 84,
    "ibm_hanoi": 89,
    "ibm_jakarta": 80,
    "ibm_kyiv": 338,
    "ibm_kyoto": 365,
    "ibm_lagos": 83,
    "ibm_lima": 74,
    "ibm_manila": 81,
    "ibm_melbourne": 65,
    "ibm_montreal": 28,
    "ibm_nairobi": 310,
    "ibm_nazca": 358,
    "ibm_oslo": 311,
    "ibm_perth": 82,
    "ibm_prague": 132,
    "ibm_sherbrooke": 363,
    "ibm_tokyo": 24,
    "ibm_torino": 366,
    "ibm_toronto": 14,
    "ibm_quito": 79,
    "qasm_simulator": 308,
    # Quantinuum
    "H1": 150,
    # Rigetti
    "aspen_9": 303,
    "aspen_11": 11,
    # IonQ
}


# Task name to Metriq ID mappings:
tasks = {
    "amplitude_estimation": 176,
    "bernstein_vazirani": 150,
    "vqe": 179,
    "hamiltonian_simulation": 178,
    "monte_carlo": 177,
    "shors": 175,
    "phase_estimation": 174,
    "hidden_shift": 173,
    "deutsch_jozsa": 172,
    "quantum_fourier_transform": 142,
    "grovers": 97,
}
