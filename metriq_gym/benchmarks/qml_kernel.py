import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import unitary_overlap
from typing import TYPE_CHECKING

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob
    from qbraid.runtime.result_data import MeasCount


@dataclass
class QMLKernelData(BenchmarkData):
    pass


class QMLKernelResult(BenchmarkResult):
    accuracy_score: BenchmarkScore

    def compute_score(self) -> float | None:
        return self.accuracy_score.value


def ZZfeature_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a ZZ feature map in the same flavor as arXiv:2405.09724.

    Args:
        num_qubits: Number of qubits

    Returns:
        A parametrized quantum kernel circuit of num_qubits qubits
    """
    layer1 = [(kk, kk + 1) for kk in range(0, num_qubits - 1, 2)]
    layer2 = [(kk, kk + 1) for kk in range(1, num_qubits - 1, 2)]

    xvec = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits)

    # Apply Hadamard gates to all qubits
    qc.h(range(num_qubits))

    # Apply Rz rotations parameterized by xvec
    for idx, param in enumerate(xvec):
        qc.rz(param, idx)

    # Apply entangling operations for both layers
    for pair in layer1 + layer2:
        var = (np.pi - xvec[pair[0]]) * (np.pi - xvec[pair[1]])
        qc.cx(pair[0], pair[1])
        qc.rz(var, pair[1])
        qc.cx(pair[0], pair[1])

    return qc


def create_inner_product_circuit(num_qubits: int, seed: int = 0) -> QuantumCircuit:
    np.random.seed(seed)

    # Create the ZZ feature map circuit and build the inner-product circuit.
    qc_qml = ZZfeature_circuit(num_qubits)
    inner_prod = unitary_overlap(qc_qml, qc_qml, insert_barrier=True)
    inner_prod.measure_all()

    # Assign parameters: using the same parameters for both copies gives perfect overlap.
    # Here we tile a random parameter vector for half the total parameters.
    param_vec = np.tile(2 * np.pi * np.random.random(size=inner_prod.num_parameters // 2), 2)
    return inner_prod.assign_parameters(param_vec)


def calculate_accuracy_score(num_qubits: int, count_results: "MeasCount") -> list[float]:
    expected_state = "0" * num_qubits
    accuracy_score = count_results.get(expected_state, 0) / sum(count_results.values())
    return [
        accuracy_score,
        np.sqrt(accuracy_score * (1 - accuracy_score) / sum(count_results.values())),
    ]


class QMLKernel(Benchmark):
    def _build_circuits(self, device: "QuantumDevice") -> QuantumCircuit:
        """Shared circuit construction logic.

        Args:
            device: The quantum device to build circuits for.

        Returns:
            The QML kernel inner product circuit.
        """
        return create_inner_product_circuit(self.params.num_qubits)

    def dispatch_handler(self, device: "QuantumDevice") -> QMLKernelData:
        circuit = self._build_circuits(device)
        return QMLKernelData.from_quantum_job(device.run(circuit, shots=self.params.shots))

    def poll_handler(
        self,
        job_data: QMLKernelData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> QMLKernelResult:
        metrics = calculate_accuracy_score(self.params.num_qubits, flatten_counts(result_data)[0])
        return QMLKernelResult(
            accuracy_score=BenchmarkScore(
                value=metrics[0],
                uncertainty=metrics[1],
            )
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list["CircuitBatch"]:
        circuit = self._build_circuits(device)
        return [CircuitBatch(circuits=[circuit], shots=self.params.shots)]
