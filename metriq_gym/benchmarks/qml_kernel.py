import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import unitary_overlap

from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts, flatten_job_ids


@dataclass
class QMLKernelData(BenchmarkData):
    pass


@dataclass
class QMLKernelResult(BenchmarkResult):
    accuracy_score: float


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


def calculate_accuracy_score(num_qubits: int, count_results: MeasCount) -> float:
    expected_state = "0" * num_qubits
    return count_results.get(expected_state, 0) / sum(count_results.values())


class QMLKernel(Benchmark):
    def dispatch_handler(self, device: QuantumDevice) -> QMLKernelData:
        qc = create_inner_product_circuit(self.params.num_qubits)
        quantum_job: QuantumJob | list[QuantumJob] = device.run(qc, shots=self.params.shots)
        provider_job_ids = flatten_job_ids(quantum_job)
        return QMLKernelData(
            provider_job_ids=provider_job_ids,
        )

    def poll_handler(
        self,
        job_data: QMLKernelData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> QMLKernelResult:
        return QMLKernelResult(
            accuracy_score=calculate_accuracy_score(
                self.params.num_qubits, flatten_counts(result_data)[0]
            )
        )
