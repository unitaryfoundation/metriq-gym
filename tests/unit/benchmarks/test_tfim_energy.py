from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qbraid.runtime.result_data import GateModelResultData, MeasCount

from metriq_gym.benchmarks.tfim_energy import (
    TFIMEnergy,
    TFIMEnergyData,
    analyze_tfim_counts,
    build_measurement_circuits,
    pauli_expectation_from_counts,
    reference_energies,
)


def test_reference_energies_match_afriqbench_baseline():
    exact, variational = reference_energies()

    assert exact == pytest.approx(-4.7587704831, abs=1e-9)
    assert variational == pytest.approx(-4.7575478600, abs=1e-8)
    assert abs(exact - variational) < 0.002


def test_measurement_circuit_count_and_order():
    circuits, labels, coefficients, qubits = build_measurement_circuits()

    assert len(circuits) == 7
    assert labels == [
        "Z0Z1",
        "Z1Z2",
        "Z2Z3",
        "X0",
        "X1",
        "X2",
        "X3",
    ]
    assert coefficients == [-1.0] * 7
    assert qubits == [[0, 1], [1, 2], [2, 3], [0], [1], [2], [3]]


def test_pauli_expectation_from_deterministic_counts():
    value, uncertainty = pauli_expectation_from_counts(
        {"0000": 100},
        [0, 1],
    )

    assert value == pytest.approx(1.0)
    assert uncertainty == pytest.approx(0.0)


def test_pauli_expectation_uses_qubit_order_consistently():
    value, _ = pauli_expectation_from_counts(
        {"0001": 100},
        [0],
    )

    assert value == pytest.approx(-1.0)


def test_analysis_separates_execution_and_application_error():
    exact, variational = reference_energies()
    data = TFIMEnergyData(
        provider_job_ids=["job"],
        shots=100,
        exact_ground_energy=exact,
        ideal_variational_energy=variational,
        term_labels=[
            "Z0Z1",
            "Z1Z2",
            "Z2Z3",
            "X0",
            "X1",
            "X2",
            "X3",
        ],
        term_coefficients=[-1.0] * 7,
        measured_qubits=[[0, 1], [1, 2], [2, 3], [0], [1], [2], [3]],
    )

    counts = [MeasCount({"0000": 100}) for _ in range(7)]
    result = analyze_tfim_counts(data, counts)

    assert result.energy.value == pytest.approx(-7.0)
    assert result.energy.uncertainty == pytest.approx(0.0)
    assert result.execution_energy_error == pytest.approx(
        abs(-7.0 - variational)
    )
    assert result.application_energy_error == pytest.approx(
        abs(-7.0 - exact)
    )
    assert 0.0 <= result.score.value <= 1.0


def test_poll_handles_batched_provider_results():
    exact, variational = reference_energies()
    data = TFIMEnergyData(
        provider_job_ids=["job"],
        shots=100,
        exact_ground_energy=exact,
        ideal_variational_energy=variational,
        term_labels=[
            "Z0Z1",
            "Z1Z2",
            "Z2Z3",
            "X0",
            "X1",
            "X2",
            "X3",
        ],
        term_coefficients=[-1.0] * 7,
        measured_qubits=[[0, 1], [1, 2], [2, 3], [0], [1], [2], [3]],
    )

    result_data = [
        GateModelResultData(
            measurement_counts=[
                MeasCount({"0000": 100})
                for _ in range(7)
            ]
        )
    ]

    benchmark = TFIMEnergy(
        args=MagicMock(),
        params=SimpleNamespace(shots=100),
    )
    result = benchmark.poll_handler(
        data,
        result_data,
        quantum_jobs=[MagicMock()],
    )

    assert result.energy.value == pytest.approx(-7.0)
