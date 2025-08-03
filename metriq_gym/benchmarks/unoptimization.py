"""Quantum circuit unoptimiation benchmark for the Metriq Gym

The quantum circuit unoptimization benchmark is based on the following paper:
    X.

A generalized version of the unoptimization protocol for error mitigation
software can also be found as a companion [software repository](X) to the above paper.
"""

import random
import warnings
from typing import Any

import numpy as np
from dataclasses import dataclass

from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
    Decompose,
    UnrollCustomDefinitions,
)

from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult


def unoptimize_circuit(
    qc: QuantumCircuit,
    iterations: int = 1,
    strategy: str = "concatenated",
    decomposition_method: str = "default",
) -> QuantumCircuit:
    """Apply the elementary recipe to a quantum circuit multiple times.

    Args:
        qc: The input quantum circuit.
        iterations: The number of times to apply the recipe.
        strategy: The strategy used in gate insertion. Options are "concatenated" or "random".

    Returns:
        new_qc: The quantum circuit after applying the recipe.
    """
    new_qc = qc.copy()
    for _ in range(iterations):
        # Step 1: Gate Insertion:
        new_qc, B1_info = insert(new_qc, strategy)

        # Step 2: Gate Swapping:
        new_qc = swap(new_qc, B1_info)

        # Step 3: Decomposition:
        new_qc = decompose(new_qc, method = decomposition_method)

        # Step 4: Synthesis:
        new_qc = synthesize(new_qc)

    return new_qc


def insert(qc: QuantumCircuit, strategy: str = "concatenated") -> QuantumCircuit:
    """Insert a two-qubit gate A and its Hermitian conjugate A† between two gates B1 and B2.

    Args:
        qc: The input quantum circuit.
        strategy: The strategy to select the pair of two-qubit gates. Options are "concatenated" or "random".

    Returns:
        new_qc: The modified quantum circuit with A and A† inserted.
        B1_info: Information about gate B1 (index, qubits, gate).
    """
    # Collect all two-qubit gates with their indices and qubits
    two_qubit_gates: list[QuantumCircuit] = []

    for idx, instruction in enumerate(qc.data):
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        if len(qargs) == 2:
            qubit_indices = [qc.find_bit(qarg).index for qarg in qargs]
            two_qubit_gates.append({"index": idx, "qubits": qubit_indices, "gate": instr})

    found_pair = False
    B1_idx = B1_qubits = B1_gate = shared_qubit = None

    if strategy == "concatenated":
        # Strategy concatenated: Find a pair of gates that share a common qubit
        for i in range(len(two_qubit_gates)):
            for j in range(i + 1, len(two_qubit_gates)):
                qubits_i = set(two_qubit_gates[i]["qubits"])
                qubits_j = set(two_qubit_gates[j]["qubits"])
                common_qubits = qubits_i & qubits_j

                if len(common_qubits) == 1:
                    B1_idx = two_qubit_gates[i]["index"]
                    B1_qubits = two_qubit_gates[i]["qubits"]
                    B1_gate = two_qubit_gates[i]["gate"]
                    shared_qubit = list(common_qubits)[0]
                    found_pair = True
                    break

            if found_pair:
                break

    elif strategy == "random":
        # Strategy random: Randomly select a two-qubit gate as B1
        if two_qubit_gates:
            gate_info = random.choice(two_qubit_gates)
            B1_idx = gate_info["index"]
            B1_qubits = gate_info["qubits"]
            B1_gate = gate_info["gate"]
            shared_qubit = B1_qubits[0]  # Choose the first qubit as shared
            found_pair = True
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Available strategies are 'concatenated' and 'random'.")

    if not found_pair or B1_idx is None or B1_qubits is None:
        warnings.warn("No suitable pair of two-qubit gates found. Skipping gate insertion.")
        return qc, None  # Return the original circuit unmodified

    # Generate a random two-qubit unitary A and its adjoint A†
    A = random_unitary(4)
    A_dag = A.adjoint()

    if B1_qubits is None:
        warnings.warn("B1_qubits is None. Skipping gate insertion.")
        return qc, None  # Return the original circuit unmodified

    # Choose qubits for A and A† insertion
    all_qubits = set(range(qc.num_qubits))
    other_qubits = list(all_qubits - set(B1_qubits))

    if not other_qubits:
        warnings.warn("Not enough qubits to perform gate insertion. Skipping.")
        return qc, None  # Return the original circuit unmodified

    third_qubit = other_qubits[0]
    if shared_qubit is None:
        warnings.warn("Shared qubit is None. Skipping gate insertion.")
        return qc, None  # Return the original circuit unmodified

    # Map indices back to qubits
    qubit_map = {qc.find_bit(q).index: q for q in qc.qubits}

    # Create a new circuit and insert A and A†
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)

    # Copy the gates up to and including B1
    for instruction in qc.data[: B1_idx + 1]:
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        new_qc.append(instr, qargs, cargs)

    # Insert A on qubits [shared_qubit, third_qubit]
    qubits_for_A = [qubit_map[shared_qubit], qubit_map[third_qubit]]

    # Insert A†, A on the same qubits
    new_qc.unitary(A_dag, qubits_for_A, label=r"$A^{\dagger}$")
    new_qc.unitary(A, qubits_for_A, label="A")

    # Copy the remaining gates
    for instruction in qc.data[B1_idx + 1 :]:
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        new_qc.append(instr, qargs, cargs)

    # Prepare B1_info for gate_swap function.
    B1_info = {
        "index": B1_idx,
        "qubits": B1_qubits,
        "gate": B1_gate,
        "shared_qubit": shared_qubit,
        "third_qubit": third_qubit,
        "A": A,
        "A_dag": A_dag,
    }

    return new_qc, B1_info


def swap(qc: QuantumCircuit, B1_info: dict[str, Any]) -> QuantumCircuit:
    r"""Swap the B1 gate with the A† gate in the circuit, replacing A† with \widetilde{A^\dagger}.

    Args:
        qc: The input quantum circuit.
        B1_info: Information about gate B1, including its index, qubits, and the A, A† gates.

    Returns:
        The modified quantum circuit with B1 and A† swapped.
    """
    B1_idx = B1_info["index"]
    B1_qubits = B1_info["qubits"]
    B1_gate = B1_info["gate"]
    A = B1_info["A"]
    shared_qubit = B1_info["shared_qubit"]
    third_qubit = B1_info["third_qubit"]

    # Map qubit indices to qubit objects.
    qubit_map = {qc.find_bit(q).index: q for q in qc.qubits}

    # Get the operators.
    B1_operator = Operator(B1_gate)
    A_operator = Operator(A)
    A_dagger_operator = A_operator.adjoint()

    # Determine the qubits involved.
    qubits_involved = sorted(set(B1_qubits + [shared_qubit, third_qubit]))
    qubits_involved_objs = [qubit_map[q] for q in qubits_involved]
    num_qubits_involved = len(qubits_involved)

    # Create mapping from qubit indices to positions.
    qubit_positions = {q: idx for idx, q in enumerate(qubits_involved)}

    # Build B1_operator_full.
    B1_operator_full = Operator(np.eye(2**num_qubits_involved))
    B1_qubit_positions = [qubit_positions[q] for q in B1_qubits]
    B1_operator_full = B1_operator_full.compose(B1_operator, qargs=B1_qubit_positions)

    # Build A_dagger_operator_full.
    A_dagger_operator_full = Operator(np.eye(2**num_qubits_involved))
    A_dagger_qubits = [shared_qubit, third_qubit]
    A_dagger_qubit_positions = [qubit_positions[q] for q in A_dagger_qubits]
    A_dagger_operator_full = A_dagger_operator_full.compose(A_dagger_operator, qargs=A_dagger_qubit_positions)

    # Compute B1_operator_full_dagger.
    B1_operator_full_dagger = B1_operator_full.adjoint()

    # Compute \widetilde{A^\dagger}.
    widetilde_A_dagger_operator = B1_operator_full_dagger.dot(A_dagger_operator_full).dot(B1_operator_full)

    # Create UnitaryGate from \widetilde{A^\dagger}.
    widetilde_A_dagger_gate = UnitaryGate(widetilde_A_dagger_operator.data, label=r"$\widetilde{A^{\dagger}}$")

    # Create a new quantum circuit.
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)

    # Copy the gates up to B1_idx.
    for i in range(B1_idx):
        instruction = qc.data[i]
        new_qc.append(instruction.operation, instruction.qubits, instruction.clbits)

    # Insert \widetilde{A^\dagger} at position B1_idx.
    new_qc.append(widetilde_A_dagger_gate, qubits_involved_objs)

    # Insert B1 gate at position B1_idx + 1.
    new_qc.append(B1_gate, [qubit_map[q] for q in B1_qubits])

    # Copy the remaining gates, skipping the original A_dagger gate.
    for i in range(B1_idx + 2, len(qc.data)):
        instruction = qc.data[i]
        new_qc.append(instruction.operation, instruction.qubits, instruction.clbits)

    return new_qc


def decompose(qc: QuantumCircuit, method: str = "default") -> QuantumCircuit:
    """Decompose multi-qubit unitary gates into elementary gates.

    Args:
        qc: The quantum circuit to decompose.
        method: The decomposition method to use. Options include:
                - "default": Standard Qiskit decomposition.
                - "kak": Perform KAK decomposition for two-qubit gates.

    Returns:
        The decomposed quantum circuit.
    """
    if method == "default":
        # Built-in default decomposition.
        return qc.decompose()

    elif method == "kak":
        # Use Decompose pass for generic decomposition.
        pass_manager = PassManager()
        pass_manager.append(Decompose())
        return pass_manager.run(qc)

    elif method == "basis":
        # Decompose using a specific target basis set.
        basis_gates = ["cx", "u3"]
        pass_manager = PassManager()
        pass_manager.append(UnrollCustomDefinitions(SessionEquivalenceLibrary, basis_gates))
        pass_manager.append(BasisTranslator(SessionEquivalenceLibrary, basis_gates))
        return pass_manager.run(qc)

    else:
        raise ValueError(f"Unknown decomposition method: {method}")


def synthesize(qc: QuantumCircuit, optimization_level: int = 3) -> QuantumCircuit:
    """Synthesize the circuit using a specified optimization level.

    Args:
        qc: The quantum circuit to synthesize.
        optimization_level: The optimization level for transpilation.

    Returns:
        The synthesized quantum circuit.
    """
    return transpile(qc, optimization_level=optimization_level, basis_gates=["cx", "u3"])


def calculate_peak_probability(shots: int, count_results: MeasCount) -> tuple[float, str]:
    """Calculate the peaked probability of a quantum circuit."""
    if not count_results:
        return 0.0, None

    max_count_outcome: str = max(count_results, key=count_results.get)
    expectation_value: float = count_results[max_count_outcome] / shots

    return expectation_value, max_count_outcome


class UnoptimizationResult(BenchmarkResult):
    """Result class to store Unoptimization benchmark results.

    Attributes:
        expectation_value: Expectation value of the unoptimized circuit.
        max_count_outcome: The bitstring for which the peaked outcome was obtained.
    """

    expectation_value: float
    max_count_outcome: str


@dataclass
class UnoptimizationData(BenchmarkData):
    """Dataclass to store Unoptimization benchmark metadata."""

    pass


class Unoptimization(Benchmark):
    """Benchmark class for Unoptimization experiments."""

    def dispatch_handler(self, device: QuantumDevice) -> UnoptimizationData:
        qc = QuantumCircuit(self.params.num_qubits)
        qc.h(0)
        for i in range(self.params.num_qubits - 1):
            qc.cx(i, i + 1)
        qc.h(0)

        qc_measured = qc.copy()
        qc_measured.measure_all()

        unopt_qc = unoptimize_circuit(
            qc,
            iterations=self.params.iterations,
            strategy=self.params.strategy,
            decomposition_method=self.params.decomposition_method,
        )
        unopt_qc_measured = unopt_qc.copy()
        unopt_qc_measured.measure_all()

        return UnoptimizationData.from_quantum_job(
            device.run(unopt_qc_measured, shots=self.params.shots)
        )

    def poll_handler(
        self,
        job_data: UnoptimizationData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> UnoptimizationResult:
        """Poll results for Unoptimization benchmark."""
        expectation_value, max_count_outcome = calculate_peak_probability(
            self.params.shots, flatten_counts(result_data)[0]
        )
        return UnoptimizationResult(
            expectation_value=expectation_value,
            max_count_outcome=max_count_outcome,
        )
