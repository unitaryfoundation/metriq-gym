"""WIT (Wormhole-inspired teleportation) benchmark for the Metriq Gym
(credit to Paul Nation for the original code for IBM devices).

The WIT benchmark is based on the following paper:
    Towards Quantum Gravity in the Lab on Quantum Processors
    Illya Shapoval, Vincent Paul Su, Wibe de Jong, Miro Urbanek, Brian Swingle
    Quantum 7, 1138 (2023)

A generalized version of the WIT benchmark software can also be found as a companion [software
repository](https://gitlab.com/ishapova/qglab/-/blob/master/scripts/wormhole.py) to the above paper.
"""

import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult


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


def calculate_expectation_value(shots: int, count_results: MeasCount) -> float:
    """Calculate the expectation value of the Pauli operator in the state produced by the quantum circuit."""
    return count_results["1"] / shots


class WITResult(BenchmarkResult):
    """Result class to store WIT benchmark results.

    Attributes:
        expectation_value: Expectation value of the Pauli operator in the state produced by the quantum circuit.
    """

    expectation_value: float


@dataclass
class WITData(BenchmarkData):
    """Dataclass to store WIT benchmark metadata."""

    pass


class WIT(Benchmark):
    """Benchmark class for WIT experiments."""

    def dispatch_handler(self, device: QuantumDevice) -> WITData:
        return WITData.from_quantum_job(
            device.run(wit_circuit(self.params.num_qubits), shots=self.params.shots)
        )

    def poll_handler(
        self,
        job_data: WITData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> WITResult:
        """Poll results for WIT benchmark."""
        return WITResult(
            expectation_value=calculate_expectation_value(
                self.params.shots, flatten_counts(result_data)[0]
            )
        )
