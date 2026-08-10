"""QSC (Quantum Simulation Circuits) benchmark implementation.

Summary:
    Runs a quantum dynamical simulation circuit based on Hamiltonian time evolution
    using Trotter decomposition, and reports per-qubit Pauli expectation values.

    The circuit consists of three stages:
    1. Initial state preparation (ground state of the Hamiltonian)
    2. A Pauli-basis perturbation on the first qubit
    3. Trotterized time evolution under the Hamiltonian
    4. Measurement in the specified Pauli basis

Result interpretation:
    Polling returns QSCResult with expectation values for each qubit measured in the
    specified Pauli basis. The output is a 2D array of real numbers (expectation values
    and standard errors) indexed by qubit.

    NOTE: The output format is preliminary and subject to change as the QSC algorithm
    is further developed.

References:
    - [Childs, Maslov, Nam, Ross, Su, "Toward the first quantum simulation with
      quantum speedup", PNAS 115(38), 2018](https://arxiv.org/abs/1912.06076)
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.statistics import (
    binary_expectation_stddev,
    binary_expectation_value,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


# ---------------------------------------------------------------------------
# Circuit construction helpers (adapted from QSC-Applications)
# ---------------------------------------------------------------------------

# Basis gates used for transpilation of QSC circuits.
_QSC_BASIS_GATES = [
    "ryy", "rxx", "rzz", "cx", "cz",
    "h", "s", "u3", "sdg", "t", "tdg",
    "ry", "rz", "rx",
]


def _generate_hamiltonian(
    num_qubits: int,
    delta: float,
    j_coupling: float,
    hx: float,
) -> SparsePauliOp:
    """Build a 1-D Heisenberg-like Hamiltonian as a SparsePauliOp.

    H = sum_{i} [ Delta*J * Z_i Z_{i+1} + J * (X_i X_{i+1} + Y_i Y_{i+1}) ] + hx * sum_i X_i

    Args:
        num_qubits: Number of lattice sites (qubits).
        delta: Anisotropy parameter (multiplies ZZ coupling).
        j_coupling: Exchange coupling strength.
        hx: Transverse field strength.

    Returns:
        The Hamiltonian as a Qiskit SparsePauliOp.
    """
    paulis: list[str] = []
    coefficients: list[float] = []
    for i in range(num_qubits - 1):
        for pauli_char, coeff in [("Z", delta * j_coupling), ("X", j_coupling), ("Y", j_coupling)]:
            label = ["I"] * num_qubits
            label[i] = pauli_char
            label[i + 1] = pauli_char
            paulis.append("".join(label))
            coefficients.append(coeff)
    for i in range(num_qubits):
        label = ["I"] * num_qubits
        label[i] = "X"
        paulis.append("".join(label))
        coefficients.append(hx)
    return SparsePauliOp(paulis, coefficients)


def _build_qsc_circuit(
    num_qubits: int,
    delta: float,
    j_coupling: float,
    hx: float,
    trotter_order: int,
    trotter_steps: int,
    evolution_time: float,
    pauli_basis: str,
) -> QuantumCircuit:
    """Construct the full QSC benchmark circuit.

    Steps:
    1. Build Hamiltonian and compute ground state via exact diagonalization.
    2. Prepare the ground state on the circuit.
    3. Apply a single-qubit Pauli-basis perturbation on qubit 0.
    4. Append Trotterized time evolution.
    5. Rotate into the measurement basis and measure all qubits.

    Args:
        num_qubits: Number of qubits / lattice sites.
        delta: Anisotropy parameter.
        j_coupling: Exchange coupling.
        hx: Transverse field.
        trotter_order: Suzuki-Trotter decomposition order.
        trotter_steps: Number of Trotter repetitions.
        evolution_time: Duration of Hamiltonian evolution.
        pauli_basis: Measurement / perturbation basis ("X", "Y", or "Z").

    Returns:
        A fully-constructed QuantumCircuit with measurements on all qubits.
    """
    hamiltonian = _generate_hamiltonian(num_qubits, delta, j_coupling, hx)

    # --- Ground state preparation ---
    h_matrix = hamiltonian.to_matrix()
    _, evecs = np.linalg.eigh(h_matrix)
    psi0 = list(evecs[:, 0])
    psi0.reverse()

    qc_init = QuantumCircuit(num_qubits)
    state = Statevector(psi0)
    qc_init.initialize(state.data, qc_init.qubits)
    qc_init = qc_init.decompose().decompose().decompose().decompose()
    qc_init.barrier()
    qc_init = transpile(qc_init, basis_gates=_QSC_BASIS_GATES, optimization_level=0)

    # --- Perturbation on qubit 0 ---
    if pauli_basis == "X":
        qc_init.rx(math.pi / 2.0, 0)
    elif pauli_basis == "Y":
        qc_init.ry(math.pi / 2.0, 0)
    elif pauli_basis == "Z":
        qc_init.rz(math.pi / 2.0, 0)
    qc_init.barrier()

    # --- Time evolution ---
    qc_evolve = QuantumCircuit(num_qubits)
    synth = SuzukiTrotter(order=trotter_order, reps=trotter_steps)
    gate = PauliEvolutionGate(hamiltonian, time=-1 * evolution_time, synthesis=synth)
    qc_evolve.append(gate, qc_evolve.qubits)
    qc_evolve = qc_evolve.decompose().decompose().decompose().decompose()
    qc_evolve = transpile(qc_evolve, basis_gates=_QSC_BASIS_GATES, optimization_level=0)

    # --- Compose and add measurement basis rotation ---
    final_circuit = qc_init.compose(qc_evolve)
    final_circuit.barrier()

    if pauli_basis == "X":
        for idx in range(num_qubits):
            final_circuit.h(idx)
    elif pauli_basis == "Y":
        for idx in range(num_qubits):
            final_circuit.sdg(idx)
            final_circuit.h(idx)
    # Z basis: no rotation needed

    final_circuit.measure_all()
    return final_circuit


# ---------------------------------------------------------------------------
# Benchmark classes
# ---------------------------------------------------------------------------


class QSCResult(BenchmarkResult):
    """Results from a QSC benchmark run.

    Attributes:
        expectation_values: Per-qubit expectation values in the measurement basis.
            Each entry is a BenchmarkScore with value and uncertainty (standard error).
        pauli_basis: The Pauli basis used for measurement ("X", "Y", or "Z").

    TODO: The output format is preliminary. Elijah is still finalizing the exact
    output structure. This may evolve to include multiple Pauli bases per run
    and/or a 2D array format.
    """

    expectation_values: list[BenchmarkScore]
    pauli_basis: str

    def compute_score(self) -> BenchmarkScore | None:
        """Return the mean expectation value across all qubits as the overall score."""
        if not self.expectation_values:
            return None
        mean_val = sum(ev.value for ev in self.expectation_values) / len(self.expectation_values)
        uncertainties = [ev.uncertainty for ev in self.expectation_values if ev.uncertainty is not None]
        mean_unc = (sum(u for u in uncertainties) / len(uncertainties)) if uncertainties else None
        return BenchmarkScore(value=mean_val, uncertainty=mean_unc)


@dataclass
class QSCData(BenchmarkData):
    """Intermediate data stored between dispatch and poll phases.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        shots: Number of measurement shots used.
        pauli_basis: Pauli basis used for measurement.
    """

    num_qubits: int = 0
    shots: int = 0
    pauli_basis: str = "Z"


class QSC(Benchmark):
    """QSC (Quantum Simulation Circuits) benchmark.

    Constructs a Hamiltonian time-evolution circuit with ground-state preparation,
    a single-qubit perturbation, Trotterized dynamics, and Pauli-basis measurement.
    Reports per-qubit expectation values.
    """

    def _build_circuit(self, device: "QuantumDevice") -> QuantumCircuit:
        """Build the QSC circuit from configured parameters."""
        return _build_qsc_circuit(
            num_qubits=self.params.num_qubits,
            delta=self.params.delta,
            j_coupling=self.params.j_coupling,
            hx=self.params.hx,
            trotter_order=self.params.trotter_order,
            trotter_steps=self.params.trotter_steps,
            evolution_time=self.params.evolution_time,
            pauli_basis=self.params.pauli_basis,
        )

    def dispatch_handler(self, device: "QuantumDevice") -> QSCData:
        circuit = self._build_circuit(device)
        quantum_job = device.run(circuit, shots=self.params.shots)
        return QSCData.from_quantum_job(
            quantum_job,
            num_qubits=self.params.num_qubits,
            shots=self.params.shots,
            pauli_basis=self.params.pauli_basis,
        )

    def poll_handler(
        self,
        job_data: QSCData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> QSCResult:
        counts = flatten_counts(result_data)[0]
        num_qubits = job_data.num_qubits
        shots = job_data.shots

        # Compute per-qubit expectation values from the bitstring counts.
        # For each qubit, marginalize the counts to that qubit and compute
        # the binary expectation value (<Z> in the computational basis,
        # which corresponds to <pauli_basis> due to the basis rotation).
        expectation_values: list[BenchmarkScore] = []
        for qubit_idx in range(num_qubits):
            # Marginalize: sum counts where qubit_idx is 0 vs 1
            counts_0 = 0
            counts_1 = 0
            for bitstring, count in counts.items():
                # Qiskit bitstrings are little-endian: rightmost bit is qubit 0
                bit = bitstring[-(qubit_idx + 1)]
                if bit == "0":
                    counts_0 += count
                else:
                    counts_1 += count
            marginalized = {"0": counts_0, "1": counts_1}
            exp_val = binary_expectation_value(shots, marginalized)
            exp_std = binary_expectation_stddev(shots, marginalized)
            expectation_values.append(BenchmarkScore(value=exp_val, uncertainty=exp_std))

        return QSCResult(
            expectation_values=expectation_values,
            pauli_basis=job_data.pauli_basis,
        )

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list[CircuitBatch]:
        circuit = self._build_circuit(device)
        return [CircuitBatch(circuits=[circuit], shots=self.params.shots)]
