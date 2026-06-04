"""QAT Operator Loschmidt Echo (OLE) benchmark implementation.

Summary:
    Loads a pre-compiled circuit from the Quantum Advantage Tracker and estimates
    the Operator Loschmidt Echo (OLE) by measuring ⟨Z_{q_0} ⊗ Z_{q_1} ⊗ Z_{q_2}⟩.

Result interpretation:
    Polling returns QATOLEResult.expectation_value as a BenchmarkScore:
        - value: estimated Pauli-Z product expectation ⟨Z_{52} Z_{59} Z_{72}⟩.
          Close to 1.0 indicates low decoherence; close to 0.0 indicates heavy noise.
        - uncertainty: propagated binomial standard deviation from the parity fractions.
    The ideal (noiseless) value for this echo protocol is 1.0.

References:
    - [Quantum Advantage Tracker OLE circuits](https://github.com/quantum-advantage-tracker/
      quantum-advantage-tracker.github.io/tree/main/data/observable-estimations/
      circuit-models/operator_loschmidt_echo)
    - Algorithmiq model description:
      https://algorithmiq.fi/wp-content/uploads/2025/11/model-information-flow-complex-material-document.pdf
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit import qasm3
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

_BASE_URL = (
    "https://raw.githubusercontent.com/quantum-advantage-tracker/"
    "quantum-advantage-tracker.github.io/main/data/observable-estimations/"
    "circuit-models/operator_loschmidt_echo/"
)

# All three circuits share the same observable: O = Z_52 Z_59 Z_72
_OBSERVABLE_QUBITS = [52, 59, 72]

_CIRCUIT_FILENAMES: dict[str, str] = {
    "49Q_L3": "49Q_OLE_circuit_L_3_b_0.25_delta0.15.qasm",
    "49Q_L6": "49Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm",
    "70Q_L6": "70Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm",
}


def _fetch_qasm(circuit_name: str) -> str:
    url = _BASE_URL + _CIRCUIT_FILENAMES[circuit_name]
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        return resp.read().decode()


def _build_ole_circuit(qasm_source: str, observable_qubits: list[int]) -> QuantumCircuit:
    """Parse a QASM 3.0 source string and append measurements on the observable qubits."""
    qc = qasm3.loads(qasm_source)
    n_obs = len(observable_qubits)
    cr = ClassicalRegister(n_obs, "c")
    qc.add_register(cr)
    for i, qubit_idx in enumerate(observable_qubits):
        qc.measure(qubit_idx, cr[i])
    return qc


def _pauli_z_product_expectation(counts: dict[str, int]) -> tuple[float, float]:
    """Compute ⟨Z_{q0} ... Z_{qk}⟩ from measurement-count bitstrings.

    For a k-qubit Pauli-Z product, the bitstring parity determines the sign:
      even parity (XOR = 0) contributes +1, odd parity (XOR = 1) contributes -1.

    Returns:
        (expectation, uncertainty) where expectation = P(even) - P(odd) and
        uncertainty is the propagated binomial standard deviation.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0

    even_count = sum(
        count for bitstring, count in counts.items() if sum(int(b) for b in bitstring) % 2 == 0
    )
    p_even = even_count / total
    expectation = 2.0 * p_even - 1.0
    uncertainty = 2.0 * sqrt(p_even * (1.0 - p_even) / total)
    return expectation, uncertainty


@dataclass
class QATOLEData(BenchmarkData):
    observable_qubits: list[int]
    shots: int


class QATOLEResult(BenchmarkResult):
    expectation_value: BenchmarkScore

    def compute_score(self) -> BenchmarkScore:
        return self.expectation_value


class QATOLE(Benchmark):
    def _build_circuit(self) -> QuantumCircuit:
        qasm_source = _fetch_qasm(self.params.circuit)
        return _build_ole_circuit(qasm_source, _OBSERVABLE_QUBITS)

    def dispatch_handler(self, device: "QuantumDevice") -> QATOLEData:
        circuit = self._build_circuit()
        return QATOLEData.from_quantum_job(
            device.run(circuit, shots=self.params.shots),
            observable_qubits=_OBSERVABLE_QUBITS,
            shots=self.params.shots,
        )

    def poll_handler(
        self,
        job_data: QATOLEData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> QATOLEResult:
        counts = flatten_counts(result_data)[0]
        value, uncertainty = _pauli_z_product_expectation(counts)
        return QATOLEResult(
            expectation_value=BenchmarkScore(value=value, uncertainty=uncertainty)
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        circuit = self._build_circuit()
        return [CircuitBatch(circuits=[circuit], shots=self.params.shots)]
