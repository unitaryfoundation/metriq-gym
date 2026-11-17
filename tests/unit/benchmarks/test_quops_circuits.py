import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

from metriq_gym.benchmarks.quops.circuits import (
    build_mirror_circuits,
    build_spam_circuit,
    randomized_compilation,
    sample_quops_circuit,
    validate_shape,
)


@pytest.mark.parametrize("width,size", [(2, 8), (3, 16), (4, 17), (5, 49), (6, 204)])
def test_workload_size_and_reproducibility(width, size):
    circuit = sample_quops_circuit(width, size, np.random.default_rng(45))
    assert sum(len(instruction.qubits) for instruction in circuit.data) == size
    assert circuit == sample_quops_circuit(width, size, np.random.default_rng(45))
    assert circuit != sample_quops_circuit(width, size, np.random.default_rng(46))
    assert set(circuit.count_ops()) <= {"cx", "rx", "ry", "rz"}


@pytest.mark.parametrize("width,size", [(0, 8), (2, 0), (2, 2), (3, 8), (4, 14)])
def test_unrepresentable_shapes_fail(width, size):
    with pytest.raises(ValueError, match="QUOPS"):
        validate_shape(width, size)


def test_pairing_randomizes_odd_idle_qubit_and_direction():
    rng = np.random.default_rng(16)
    pairs = set()
    idle_qubits = set()
    for _ in range(100):
        circuit = sample_quops_circuit(3, 5, rng)
        control, target = [circuit.find_bit(q).index for q in circuit.data[0].qubits]
        pairs.add((control, target))
        idle_qubits.update({0, 1, 2} - {control, target})
    assert len(pairs) == 6
    assert idle_qubits == {0, 1, 2}


def test_partial_round_selects_random_subset():
    rng = np.random.default_rng(7)
    selected = set()
    for _ in range(100):
        circuit = sample_quops_circuit(4, 4, rng)
        selected.add(frozenset(circuit.find_bit(q).index for q in circuit.data[0].qubits))
        assert len(circuit.data) == 3  # One CNOT, two rotations.
    assert len(selected) == 6


@pytest.mark.parametrize("basis", [["u", "cx"], ["u", "cz"]])
@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_randomized_compilation_preserves_unitary_and_entanglers(basis, width):
    rng = np.random.default_rng(84)
    workload = sample_quops_circuit(width, 3 * (width + 2 * (width // 2)), rng)
    compiled = transpile(workload, basis_gates=basis, optimization_level=3)
    reference = randomized_compilation(compiled, rng)
    assert Operator(reference).equiv(Operator(compiled))
    assert reference.count_ops().get(basis[1], 0) == compiled.count_ops().get(basis[1], 0)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 6])
def test_mirrors_and_spam_have_known_ideal_outcomes(width):
    rng = np.random.default_rng(width)
    workload = sample_quops_circuit(width, 3 * (width + 2 * (width // 2)), rng)
    compiled = transpile(workload, basis_gates=["u", "cz"], optimization_level=3)
    for mirror in (*build_mirror_circuits(compiled, rng), build_spam_circuit(width, rng)):
        probabilities = Statevector.from_instruction(
            mirror.circuit.remove_final_measurements(inplace=False)
        ).probabilities_dict()
        assert probabilities[mirror.ideal_bitstring] == pytest.approx(1)
        assert len(mirror.ideal_bitstring) == width


def test_mirror_boundaries_survive_native_translation():
    rng = np.random.default_rng(17)
    compiled = transpile(
        sample_quops_circuit(4, 64, rng), basis_gates=["u", "cz"], optimization_level=3
    )
    for mirror in build_mirror_circuits(compiled, rng):
        translated = transpile(
            mirror.circuit, basis_gates=["rz", "sx", "x", "cz"], optimization_level=0
        )
        assert translated.count_ops()["cz"] == 2 * compiled.count_ops()["cz"]
        assert translated.count_ops()["barrier"] == mirror.circuit.count_ops()["barrier"]


def test_reference_rejects_non_clifford_entanglers():
    circuit = QuantumCircuit(2)
    circuit.cp(0.3, 0, 1)
    with pytest.raises(ValueError, match="native CX/CZ"):
        randomized_compilation(circuit, np.random.default_rng(9))


def test_reference_dressings_are_independent_and_reproducible():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 1)
    circuit.ry(0.7, 1)
    circuit.cz(1, 2)
    first = randomized_compilation(circuit, np.random.default_rng(8))
    assert first == randomized_compilation(circuit, np.random.default_rng(8))
    assert first != randomized_compilation(circuit, np.random.default_rng(9))


def test_reference_rejects_measured_input():
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    with pytest.raises(ValueError, match="unmeasured"):
        randomized_compilation(circuit, np.random.default_rng(9))
