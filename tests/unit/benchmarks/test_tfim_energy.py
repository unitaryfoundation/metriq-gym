from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from jsonschema.exceptions import ValidationError
from qbraid import QuantumJob
from qbraid.runtime.result_data import GateModelResultData, MeasCount

from metriq_gym.benchmarks.tfim_energy import (
    TFIMEnergy,
    TFIMEnergyData,
    TFIMEnergyResult,
    analyze_tfim_counts,
    build_measurement_circuits,
    pauli_expectation_from_counts,
    reference_energies,
)
from metriq_gym.constants import JobType, SCHEMA_MAPPING
from metriq_gym.registry import (
    BENCHMARK_DATA_CLASSES,
    BENCHMARK_HANDLERS,
    BENCHMARK_RESULT_CLASSES,
)
from metriq_gym.schema_validator import validate_and_create_model


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


def test_dispatch_submits_one_seven_circuit_batch():
    device = MagicMock()
    device.num_qubits = 5
    job = MagicMock(spec=QuantumJob)
    job.id = "tfim-test-job"
    device.run.return_value = job

    benchmark = TFIMEnergy(
        args=MagicMock(),
        params=SimpleNamespace(shots=256),
    )
    data = benchmark.dispatch_handler(device)

    device.run.assert_called_once()
    circuits = device.run.call_args.args[0]

    assert len(circuits) == 7
    assert device.run.call_args.kwargs["shots"] == 256
    assert data.provider_job_ids == ["tfim-test-job"]
    assert data.shots == 256
    assert data.input_two_qubit_gate_counts == [6] * 7
    assert data.transpiled_two_qubit_gate_counts == [6] * 7


def test_resource_estimation_matches_dispatch_shape():
    device = MagicMock()
    device.num_qubits = 5

    benchmark = TFIMEnergy(
        args=MagicMock(),
        params=SimpleNamespace(shots=512),
    )
    batches = benchmark.estimate_resources_handler(device)

    assert len(batches) == 1
    assert len(batches[0].circuits) == 7
    assert batches[0].shots == 512


def test_registry_and_schema_mapping_are_registered():
    assert JobType.TFIM_ENERGY.value == "TFIM Energy"
    assert SCHEMA_MAPPING[JobType.TFIM_ENERGY] == "tfim_energy.schema.json"
    assert BENCHMARK_HANDLERS[JobType.TFIM_ENERGY] is TFIMEnergy
    assert BENCHMARK_DATA_CLASSES[JobType.TFIM_ENERGY] is TFIMEnergyData
    assert BENCHMARK_RESULT_CLASSES[JobType.TFIM_ENERGY] is TFIMEnergyResult


def test_schema_accepts_canonical_configuration():
    params = validate_and_create_model(
        {
            "benchmark_name": "TFIM Energy",
            "num_qubits": 4,
            "coupling_j": 1.0,
            "field_h": 1.0,
            "shots": 8192,
        }
    )

    assert params.benchmark_name == "TFIM Energy"
    assert params.num_qubits == 4
    assert params.shots == 8192


def test_schema_rejects_noncanonical_field_strength():
    with pytest.raises(ValidationError):
        validate_and_create_model(
            {
                "benchmark_name": "TFIM Energy",
                "num_qubits": 4,
                "coupling_j": 1.0,
                "field_h": 1.5,
                "shots": 8192,
            }
        )
