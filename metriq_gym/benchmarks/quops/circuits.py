"""QUOPS workloads and physical-qubit mirror circuit fidelity estimation.

The circuit distribution and three mirror ensembles follow Sections II.2 and
IV.1–IV.2 of https://arxiv.org/abs/2609.12146. Randomization is applied *after*
routing, to the native Clifford entanglers, and absorbed into single-qubit gates.
"""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, random_clifford
from qiskit.synthesis import OneQubitEulerDecomposer


@dataclass
class MirrorCircuit:
    circuit: QuantumCircuit
    ideal_bitstring: str


def validate_shape(width: int, size: int) -> None:
    """Check that a shape can be sampled, including a partially filled final round."""
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("QUOPS width must be a positive integer")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("QUOPS size must be a positive integer")
    round_size = width + 2 * (width // 2)
    remainder = size % round_size
    if remainder not in {k + 2 * (k // 2) for k in range(width + 1)}:
        raise ValueError(f"QUOPS size {size} is not representable at width {width}")


def sample_quops_circuit(width: int, size: int, rng: Generator) -> QuantumCircuit:
    """Sample random CNOT pairings followed by independent Pauli-axis rotations.

    Size counts one per rotation and two per CNOT, before hardware compilation.
    The final round may act on a uniformly sampled subset of the qubits.
    """
    validate_shape(width, size)
    full_rounds, remainder = divmod(size, width + 2 * (width // 2))
    final_width = next(k for k in range(width + 1) if k + 2 * (k // 2) == remainder)
    circuit = QuantumCircuit(width, name=f"quops_{width}_{size}")
    for active_width in [width] * full_rounds + ([final_width] if final_width else []):
        # The permutation samples the subset, matching, direction, and odd idle wire.
        qubits = rng.permutation(width)[:active_width].tolist()
        for control, target in zip(qubits[::2], qubits[1::2]):
            circuit.cx(control, target)
        for qubit in qubits:
            rotation = (circuit.rx, circuit.ry, circuit.rz)[rng.integers(3)]
            rotation(rng.uniform(0, 2 * np.pi), qubit)
    return circuit


_IDENTITY: NDArray[np.complex128] = np.eye(2, dtype=complex)
# Indexed by x + 2*z, ignoring Pauli phases, which cannot affect measurement.
_PAULIS = np.array(
    [
        _IDENTITY,
        [[0, 1], [1, 0]],
        [[1, 0], [0, -1]],
        [[0, -1j], [1j, 0]],
    ],
    dtype=complex,
)
_EULER = OneQubitEulerDecomposer("U")


def _append_local_layer(circuit: QuantumCircuit, matrices: NDArray[np.complex128]) -> None:
    for qubit, matrix in enumerate(matrices):
        circuit.compose(_EULER(matrix), qubits=[qubit], inplace=True)


def _layered_circuit(circuit: QuantumCircuit, rng: Generator | None) -> QuantumCircuit:
    """Fold local gates between protected entangler layers, optionally Pauli dressing."""
    if circuit.num_clbits:
        raise ValueError("QUOPS mirroring requires an unmeasured unitary circuit")
    width = circuit.num_qubits
    result = QuantumCircuit(width)
    result.global_phase = circuit.global_phase
    pending = np.repeat(_IDENTITY[np.newaxis, :, :], width, axis=0)
    dag = circuit_to_dag(circuit)
    for layer in dag.layers():
        entanglers = []
        for node in layer["graph"].op_nodes():
            if node.op.name == "barrier":
                continue
            qubits = [dag.find_bit(qubit).index for qubit in node.qargs]
            if len(qubits) == 1 and node.op.name not in {"measure", "reset", "delay"}:
                pending[qubits[0]] = Operator(node.op).data @ pending[qubits[0]]
            elif node.op.name in {"cx", "cz"}:
                entanglers.append((node.op, qubits))
            else:
                raise ValueError(
                    "QUOPS reference compilation requires single-qubit unitaries and "
                    f"native CX/CZ gates; got {node.op.name!r}"
                )
        if not entanglers:
            continue

        if rng is not None:
            x, z = rng.integers(0, 2, size=(2, width))
            pending = _PAULIS[x + 2 * z] @ pending
        _append_local_layer(result, pending)
        result.barrier()
        for gate, qubits in entanglers:
            result.append(gate, qubits)
            if rng is not None:
                control, target = qubits
                if gate.name == "cx":
                    x[target] ^= x[control]
                    z[control] ^= z[target]
                else:
                    z[target] ^= x[control]
                    z[control] ^= x[target]
        result.barrier()
        # C P C† is a Pauli layer because every native entangler is Clifford.
        pending = (
            _PAULIS[x + 2 * z].copy()
            if rng is not None
            else np.repeat(_IDENTITY[np.newaxis, :, :], width, axis=0)
        )
    _append_local_layer(result, pending)
    return result


def randomized_compilation(circuit: QuantumCircuit, rng: Generator) -> QuantumCircuit:
    """Pauli-randomize each routed CX/CZ layer without changing its ideal unitary.

    Compensating Paulis are folded into adjacent one-qubit layers. Barriers keep
    subsequent basis translation from undoing the randomization or the mirror.
    """
    return _layered_circuit(circuit, rng)


def _cap_mirror(body: QuantumCircuit, rng: Generator, name: str) -> MirrorCircuit:
    width = body.num_qubits
    preparation = np.array([Operator(random_clifford(1, seed=rng)).data for _ in range(width)])
    x, z = rng.integers(0, 2, size=(2, width))
    readout = _PAULIS[x + 2 * z] @ preparation.conj().transpose(0, 2, 1)
    circuit = QuantumCircuit(width, name=name)
    _append_local_layer(circuit, preparation)
    circuit.barrier()
    if body.data:
        circuit.compose(body, inplace=True)
        circuit.barrier()
    _append_local_layer(circuit, readout)
    circuit.measure_all()
    return MirrorCircuit(circuit, "".join(str(bit) for bit in reversed(x)))


def build_mirror_circuits(
    compiled: QuantumCircuit, rng: Generator
) -> tuple[MirrorCircuit, MirrorCircuit]:
    """Return M1 (test/reference) and M2 (reference/reference) for one workload.

    ``compiled`` must already be routed onto exactly its computational qubits,
    expressed with arbitrary single-qubit gates and native CX/CZ entanglers.
    Each reference half is randomized independently. Ideal outputs are strings
    in Qiskit's counts order, with qubit zero at the right.
    """
    inverse = compiled.inverse()
    mixed = _layered_circuit(compiled, None)
    mixed.barrier()
    mixed.compose(randomized_compilation(inverse, rng), inplace=True)
    reference = randomized_compilation(compiled, rng)
    reference.barrier()
    reference.compose(randomized_compilation(inverse, rng), inplace=True)
    return (
        _cap_mirror(mixed, rng, "quops_mixed"),
        _cap_mirror(reference, rng, "quops_reference"),
    )


def build_spam_circuit(width: int, rng: Generator) -> MirrorCircuit:
    """Return an independently randomized M3 state-preparation/readout reference."""
    if width < 1:
        raise ValueError("QUOPS width must be positive")
    return _cap_mirror(QuantumCircuit(width), rng, "quops_spam")
