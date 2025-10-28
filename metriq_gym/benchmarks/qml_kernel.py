"""Quantum Machine Learning Kernel benchmark implementation.

Summary:
    Constructs a ZZ feature map kernel, computes the inner-product circuit, and measures the
    probability of returning to the all-zero state as a proxy for kernel quality.

Schema parameters (metriq_gym/schemas/qml_kernel.schema.json):
    - benchmark_name (str, required): must be "QML Kernel".
    - num_qubits (int, required): number of qubits in the feature map.
    - shots (int, optional, default 1000): measurement repetitions for estimating accuracy.

CLI dispatch example::

        uv run mgym job dispatch metriq_gym/schemas/examples/qml_kernel.example.json -p local -d aer_simulator

Result interpretation:
    Polling returns QMLKernelResult.accuracy_score as a BenchmarkScore where:
        - value: fraction of shots measuring the expected all-zero bitstring.
        - uncertainty: binomial standard deviation from the sample counts.
    Higher accuracy suggests better kernel reproducibility on the selected hardware.

Reference:
    - Inspired by ZZ-feature map approaches, e.g., arXiv:2405.09724.
"""

import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import unitary_overlap
from typing import TYPE_CHECKING

from pydantic import Field
from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
    MetricDirection,
)
from metriq_gym.helpers.task_helpers import flatten_counts

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob
    from qbraid.runtime.result_data import MeasCount


@dataclass
class QMLKernelData(BenchmarkData):
    pass


class QMLKernelResult(BenchmarkResult):
    accuracy_score: BenchmarkScore = Field(
        ..., json_schema_extra={"direction": MetricDirection.HIGHER}
    )


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
    def dispatch_handler(self, device: "QuantumDevice") -> QMLKernelData:
        return QMLKernelData.from_quantum_job(
            device.run(
                create_inner_product_circuit(self.params.num_qubits), shots=self.params.shots
            )
        )

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
