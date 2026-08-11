import argparse

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from metriq_gym.benchmarks.qsc import QSC, QSCData, QSCResult, _ground_state_prep_gate
from metriq_gym.schema_validator import load_and_validate

EXAMPLE = "metriq_gym/schemas/examples/qsc.example.json"


@pytest.fixture
def benchmark():
    return QSC(argparse.Namespace(), load_and_validate(EXAMPLE))


def test_ground_state_prep_gate_prepares_the_ground_state():
    from qiskit.quantum_info import SparsePauliOp

    hamiltonian = SparsePauliOp(["XXI", "IXX", "ZII", "IZI", "IIZ"], [1.0, 1.0, -2.0, -2.0, -2.0])
    gate = _ground_state_prep_gate(hamiltonian, 3)

    circuit = QuantumCircuit(3)
    circuit.append(gate, range(3))
    prepared = Statevector(circuit).data

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.to_matrix())
    overlap = abs(np.vdot(eigenvectors[:, 0], prepared)) ** 2
    assert overlap == pytest.approx(1.0, abs=1e-9)


def test_batch_size_is_twelve_circuits_per_time_step(benchmark):
    """DFS-CS emits 3 unperturbed plus 9 perturbation/measurement pairs per step."""
    circuits = benchmark._build_circuits(None)
    assert len(circuits) == 12 * benchmark.params.time_steps


def test_every_circuit_starts_from_the_ground_state(benchmark):
    """Regression guard for the upstream prep gate never being applied.

    QSC-Applications leaves `_state_prep_gate` at None, which silently starts
    every circuit from |0...0>. The benchmark injects the gate instead, so it
    must appear in every circuit it builds.
    """
    for circuit in benchmark._build_circuits(None):
        assert "ground_state_prep" in [item.operation.name for item in circuit.data]


def test_circuit_names_encode_the_experiment(benchmark):
    """QSC's post-processing keys off the circuit name, so the format matters."""
    names = [circuit.name for circuit in benchmark._build_circuits(None)]
    assert all(name.startswith("RamseyState_") for name in names)
    # site, perturbation basis, measurement basis, strength, time-step index
    assert all(len(name.split("_")) == 6 for name in names)


def test_poll_rejects_a_count_mismatch(benchmark):
    job_data = QSCData(provider_job_ids=[], sites=5, shots=10, circuit_names=["a", "b"])
    with pytest.raises(ValueError, match="expected 2 count sets"):
        benchmark.poll_handler(job_data, [], [])


def test_score_is_withheld_until_a_reference_exists():
    """G^R is only meaningful against a reference, so no scalar is reported."""
    result = QSCResult(
        green_function=[],
        time_steps=1,
        sites=2,
        perturbation_basis="X",
        measurement_basis="X",
    )
    assert result.compute_score() is None
