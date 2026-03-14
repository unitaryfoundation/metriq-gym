"""WIT (wormhole-inspired teleportation) benchmark implementation.

Summary:
    Runs a six- or seven-qubit teleportation-inspired circuit that mimics the protocol from
    Shapoval et al. (2023) and reports a Pauli-Z expectation value with binomial uncertainty.

Result interpretation:
    Polling returns WITResult.expectation_value as a BenchmarkScore:
        - value: estimated Pauli-Z expectation (ideal teleportation trends toward +1).
        - uncertainty: binomial standard deviation computed from the observed counts.
    Compare value versus uncertainty to decide whether more shots are required or if noise is
    degrading the teleportation fidelity.

References:
    - [Shapoval et al., "Towards Quantum Gravity in the Lab on Quantum Processors",
      Quantum 7, 1138 (2023)](https://arxiv.org/abs/2205.14081).
    - [Companion script](https://gitlab.com/ishapova/qglab/-/blob/master/scripts/wormhole.py).
    - Implementation lineage credited to Paul Nation (IBM Quantum).
"""

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from metriq_gym.helpers.task_helpers import flatten_counts
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
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


def wit_circuit(num_qubits: int) -> QuantumCircuit:
    """Create a WIT circuit for either 6 or 7 qubits.

    The 7-qubit circuit is based on the circuit diagram in Figure-4 of
    [arXiv:2205.14081](https://arxiv.org/pdf/2205.14081). Both the 6- and 7-qubit circuits assume a interraction
    coupling constant (referred to as `g` in the paper) of pi/2.
    """
    if num_qubits == 6:
        qc = QuantumCircuit(6, 1)
        qc.h(0)
        qc.cx(0, 5)
        qc.h(1)
        qc.cx(1, 4)
        qc.h(2)
        qc.cx(2, 3)
        qc.rx(-np.pi / 2, 0)
        qc.rx(-np.pi / 2, 1)
        qc.rx(-np.pi / 2, 2)
        qc.rz(-0.0566794, 0)
        qc.rz(-0.01039906, 1)
        qc.rz(-0.0632158, 2)
        qc.rzz(-np.pi / 2, 0, 1)
        qc.rzz(-np.pi / 2, 1, 2)
        qc.rx(-np.pi / 2, 0)
        qc.rx(-np.pi / 2, 1)
        qc.rx(-np.pi / 2, 2)
        qc.rz(-0.0566794, 0)
        qc.rz(-0.01039906, 1)
        qc.rz(-0.0632158, 2)
        qc.rzz(-np.pi / 2, 0, 1)
        qc.rzz(-np.pi / 2, 1, 2)
        qc.rx(-np.pi / 2, 0)
        qc.rx(-np.pi / 2, 1)
        qc.rx(-np.pi / 2, 2)
        qc.rz(-0.0566794, 0)
        qc.rz(-0.01039906, 1)
        qc.rz(-0.0632158, 2)
        qc.rzz(-np.pi / 2, 0, 1)
        qc.rzz(-np.pi / 2, 1, 2)
        qc.reset(0)
        qc.rz(0.0566794, 0)
        qc.rz(0.01039906, 1)
        qc.rz(0.0632158, 2)
        qc.rzz(np.pi / 2, 0, 1)
        qc.rzz(np.pi / 2, 1, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)
        qc.rx(np.pi / 2, 2)
        qc.rz(0.0566794, 0)
        qc.rz(0.01039906, 1)
        qc.rz(0.0632158, 2)
        qc.rzz(np.pi / 2, 0, 1)
        qc.rzz(np.pi / 2, 1, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)
        qc.rx(np.pi / 2, 2)
        qc.rz(0.0566794, 0)
        qc.rz(0.01039906, 1)
        qc.rz(0.0632158, 2)
        qc.rzz(np.pi / 2, 0, 1)
        qc.rzz(np.pi / 2, 1, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)
        qc.rx(np.pi / 2, 2)
        # Here are the two RZZ gates that are parameterized
        qc.rzz(np.pi / 2, 1, 4)
        qc.rzz(np.pi / 2, 2, 3)
        # -------------------------------------------------
        qc.rx(np.pi / 2, 5)
        qc.rx(np.pi / 2, 4)
        qc.rx(np.pi / 2, 3)
        qc.rz(0.0566794, 5)
        qc.rz(0.01039906, 4)
        qc.rz(0.0632158, 3)
        qc.rzz(np.pi / 2, 5, 4)
        qc.rzz(np.pi / 2, 4, 3)
        qc.rx(np.pi / 2, 5)
        qc.rx(np.pi / 2, 4)
        qc.rx(np.pi / 2, 3)
        qc.rz(0.0566794, 5)
        qc.rz(0.01039906, 4)
        qc.rz(0.0632158, 3)
        qc.rzz(np.pi / 2, 5, 4)
        qc.rzz(np.pi / 2, 4, 3)
        qc.rx(np.pi / 2, 5)
        qc.rx(np.pi / 2, 4)
        qc.rx(np.pi / 2, 3)
        qc.rz(0.0566794, 5)
        qc.rz(0.01039906, 4)
        qc.rz(0.0632158, 3)
        qc.rzz(np.pi / 2, 5, 4)
        qc.rzz(np.pi / 2, 4, 3)

        # Perform a measurement which corresponds to the Pauli-Z operator (SparsePauliOp('ZIIIII')). Since Qiskit is
        # little-endian, this is reversed, and the measurement is actually performed on the 5-th qubit.
        qc.measure(5, 0)
        return qc

    elif num_qubits == 7:
        qc = QuantumCircuit(7, 1)
        qc.h(0)
        qc.cx(0, 5)
        qc.h(1)
        qc.cx(1, 4)
        qc.h(2)
        qc.cx(2, 3)
        qc.rx(-np.pi / 2, 0)
        qc.rx(-np.pi / 2, 1)
        qc.rx(-np.pi / 2, 2)
        qc.rz(-0.0566794, 0)
        qc.rz(-0.01039906, 1)
        qc.rz(-0.0632158, 2)
        qc.rzz(-np.pi / 2, 0, 1)
        qc.rzz(-np.pi / 2, 1, 2)
        qc.rx(-np.pi / 2, 0)
        qc.rx(-np.pi / 2, 1)
        qc.rx(-np.pi / 2, 2)
        qc.rz(-0.0566794, 0)
        qc.rz(-0.01039906, 1)
        qc.rz(-0.0632158, 2)
        qc.rzz(-np.pi / 2, 0, 1)
        qc.rzz(-np.pi / 2, 1, 2)
        qc.rx(-np.pi / 2, 0)
        qc.rx(-np.pi / 2, 1)
        qc.rx(-np.pi / 2, 2)
        qc.rz(-0.0566794, 0)
        qc.rz(-0.01039906, 1)
        qc.rz(-0.0632158, 2)
        qc.rzz(-np.pi / 2, 0, 1)
        qc.rzz(-np.pi / 2, 1, 2)
        qc.swap(0, 6)
        qc.rz(0.0566794, 0)
        qc.rz(0.01039906, 1)
        qc.rz(0.0632158, 2)
        qc.rzz(np.pi / 2, 0, 1)
        qc.rzz(np.pi / 2, 1, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)
        qc.rx(np.pi / 2, 2)
        qc.rz(0.0566794, 0)
        qc.rz(0.01039906, 1)
        qc.rz(0.0632158, 2)
        qc.rzz(np.pi / 2, 0, 1)
        qc.rzz(np.pi / 2, 1, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)
        qc.rx(np.pi / 2, 2)
        qc.rz(0.0566794, 0)
        qc.rz(0.01039906, 1)
        qc.rz(0.0632158, 2)
        qc.rzz(np.pi / 2, 0, 1)
        qc.rzz(np.pi / 2, 1, 2)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)
        qc.rx(np.pi / 2, 2)
        # Here are the two RZZ gates that are parameterized
        qc.rzz(np.pi / 2, 1, 4)
        qc.rzz(np.pi / 2, 2, 3)
        # -------------------------------------------------
        qc.rx(np.pi / 2, 5)
        qc.rx(np.pi / 2, 4)
        qc.rx(np.pi / 2, 3)
        qc.rz(0.0566794, 5)
        qc.rz(0.01039906, 4)
        qc.rz(0.0632158, 3)
        qc.rzz(np.pi / 2, 5, 4)
        qc.rzz(np.pi / 2, 4, 3)
        qc.rx(np.pi / 2, 5)
        qc.rx(np.pi / 2, 4)
        qc.rx(np.pi / 2, 3)
        qc.rz(0.0566794, 5)
        qc.rz(0.01039906, 4)
        qc.rz(0.0632158, 3)
        qc.rzz(np.pi / 2, 5, 4)
        qc.rzz(np.pi / 2, 4, 3)
        qc.rx(np.pi / 2, 5)
        qc.rx(np.pi / 2, 4)
        qc.rx(np.pi / 2, 3)
        qc.rz(0.0566794, 5)
        qc.rz(0.01039906, 4)
        qc.rz(0.0632158, 3)
        qc.rzz(np.pi / 2, 5, 4)
        qc.rzz(np.pi / 2, 4, 3)

        # Perform a measurement which corresponds to the Pauli-Z operator (SparsePauliOp('IZIIIII')). Since Qiskit is
        # little-endian, this is reversed, and the measurement is actually performed on the 5-th qubit. This can also
        # be seen in Figure-4 of the paper.
        qc.measure(5, 0)

        return qc
    else:
        raise ValueError(f"Unsupported number of qubits: {num_qubits}")


class WITResult(BenchmarkResult):
    expectation_value: BenchmarkScore

    def compute_score(self) -> BenchmarkScore:
        return self.expectation_value


@dataclass
class WITData(BenchmarkData):
    pass


class WIT(Benchmark):
    def _build_circuits(self, device: "QuantumDevice") -> QuantumCircuit:
        """Shared circuit construction logic.

        Args:
            device: The quantum device to build circuits for.

        Returns:
            The WIT quantum circuit.
        """
        return wit_circuit(self.params.num_qubits)

    def dispatch_handler(self, device: "QuantumDevice") -> WITData:
        circuit = self._build_circuits(device)
        return WITData.from_quantum_job(device.run(circuit, shots=self.params.shots))

    def poll_handler(
        self,
        job_data: WITData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> WITResult:
        counts = flatten_counts(result_data)[0]
        return WITResult(
            expectation_value=BenchmarkScore(
                value=binary_expectation_value(self.params.shots, counts),
                uncertainty=binary_expectation_stddev(self.params.shots, counts),
            )
        )

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list[CircuitBatch]:
        circuit = self._build_circuits(device)
        return [CircuitBatch(circuits=[circuit], shots=self.params.shots)]
