"""QAT Operator Loschmidt Echo (OLE) benchmark implementation.

Summary:
    Loads an OLE circuit from the Quantum Advantage Tracker (by name) or from
    a local OpenQASM 3.0 file (via qasm_path), appends measurements on the
    specified observable qubits, and computes the Pauli-Z product expectation
    value ⟨Z_{q_0} ⊗ ... ⊗ Z_{q_k}⟩ = P(even parity) − P(odd parity).

Result interpretation:
    - observable_value: ⟨Z_{q_0} ... Z_{q_k}⟩ for the specific circuit and
      observable measured. This is a raw expectation value for a single
      instance; interpreting it as a decoherence or fidelity metric requires
      a reference value (e.g. noiseless simulation) that is not yet included.
    - circuit_id: name of the named circuit or basename of qasm_path.
    Score is left unset pending a reference-based definition.

Local simulator usage:
    Create an example config referencing a small fixture circuit:

        {
          "benchmark_name": "QAT OLE",
          "qasm_path": "metriq_gym/schemas/examples/qat_ole_small.qasm",
          "observable_qubits": [0, 1, 2],
          "shots": 100
        }

    qasm_path is resolved relative to the current working directory.

    Then run:

        mgym job dispatch metriq_gym/schemas/examples/qat_ole.small.example.json \\
            -p local -d aer_simulator
        mgym job poll latest

Real hardware usage:
    Use a named circuit to fetch from the Quantum Advantage Tracker:

        {
          "benchmark_name": "QAT OLE",
          "circuit": "49Q_L3",
          "shots": 1000
        }

    The named circuits are pre-compiled to 156 physical qubits and require a
    large-scale device such as IBM Eagle or IBM Heron.

References:
    - QAT OLE circuits: https://github.com/quantum-advantage-tracker/
      quantum-advantage-tracker.github.io/tree/main/data/observable-estimations/
      circuit-models/operator_loschmidt_echo
    - Algorithmiq model description:
      https://algorithmiq.fi/wp-content/uploads/2025/11/model-information-flow-complex-material-document.pdf
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

from qiskit import ClassicalRegister, QuantumCircuit
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

# Pinned to a specific upstream commit for reproducibility.
# Update this SHA (and verify the circuits) when pulling in new QAT data.
_QAT_COMMIT = "3dcd31e9aefb461fc327b58d1c2506948b9a7a3e"
_BASE_URL = (
    "https://raw.githubusercontent.com/quantum-advantage-tracker/"
    f"quantum-advantage-tracker.github.io/{_QAT_COMMIT}/data/observable-estimations/"
    "circuit-models/operator_loschmidt_echo/"
)

# Observable is O = Z_52 ⊗ Z_59 ⊗ Z_72 for all named QAT circuits.
_OBSERVABLE_QUBITS = [52, 59, 72]

_CIRCUIT_FILENAMES: dict[str, str] = {
    "49Q_L3": "49Q_OLE_circuit_L_3_b_0.25_delta0.15.qasm",
    "49Q_L6": "49Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm",
    "70Q_L6": "70Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm",
}


def _fetch_qasm(circuit_name: str) -> str:
    url = _BASE_URL + _CIRCUIT_FILENAMES[circuit_name]
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def _load_qasm_source(params) -> tuple[str, str, list[int]]:
    """Return (qasm_source, circuit_id, observable_qubits) from params."""
    circuit_name = getattr(params, "circuit", None)
    qasm_path = getattr(params, "qasm_path", None)
    obs_qubits = getattr(params, "observable_qubits", None)

    if circuit_name is not None and qasm_path is not None:
        raise ValueError("'circuit' and 'qasm_path' are mutually exclusive — set only one")
    if circuit_name is not None:
        return _fetch_qasm(circuit_name), circuit_name, _OBSERVABLE_QUBITS
    if qasm_path is not None:
        # qasm_path is resolved relative to the current working directory.
        with open(qasm_path, encoding="utf-8") as f:
            source = f.read()
        if obs_qubits is None:
            raise ValueError("observable_qubits must be set when using qasm_path")
        return source, os.path.basename(qasm_path), list(obs_qubits)
    raise ValueError("Either 'circuit' or 'qasm_path' must be specified in the config")


def _build_ole_circuit(
    qasm_source: str, observable_qubits: list[int]
) -> QuantumCircuit:
    """Parse a QASM 3.0 source string and append measurements on the observable qubits.

    Raises ValueError if the circuit already contains classical bits/measurements
    (which would make the added observable register ambiguous in the counts) or if
    any observable qubit index is out of range or duplicated.
    """
    # qiskit.qasm3 is included in qiskit >= 1.x; no separate package needed.
    qc = qasm3.loads(qasm_source)

    if qc.num_clbits > 0:
        raise ValueError(
            "Input QASM circuit already contains classical bits or measurements. "
            "Provide a circuit with no classical registers so the observable "
            "register can be unambiguously parsed from the counts."
        )

    num_qubits = qc.num_qubits
    if not observable_qubits:
        raise ValueError("observable_qubits must contain at least one index")
    if len(observable_qubits) != len(set(observable_qubits)):
        raise ValueError(f"observable_qubits contains duplicates: {observable_qubits}")
    out_of_range = [q for q in observable_qubits if not (0 <= q < num_qubits)]
    if out_of_range:
        raise ValueError(
            f"observable_qubits {out_of_range} are out of range for a "
            f"{num_qubits}-qubit circuit"
        )

    cr = ClassicalRegister(len(observable_qubits), "c")
    qc.add_register(cr)
    for i, qubit_idx in enumerate(observable_qubits):
        qc.measure(qubit_idx, cr[i])
    return qc


def _pauli_z_product_expectation(counts: dict[str, int]) -> tuple[float, float]:
    """Compute ⟨Z_{q0} ... Z_{qk}⟩ from measurement-count bitstrings.

    Even-parity outcomes contribute +1, odd-parity outcomes contribute −1.
    Returns (expectation, uncertainty) where uncertainty is the propagated
    binomial standard deviation (2σ from the parity fraction).

    Note: this uncertainty reflects shot noise for a single circuit run only.
    When initial-state averaging (N_init sampling) is added, the reported
    uncertainty will need to combine both contributions.
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
    circuit_id: str
    num_qubits: int
    num_gates: int


class QATOLEResult(BenchmarkResult):
    observable_value: BenchmarkScore
    circuit_id: str

    # score is intentionally left unset (compute_score returns None from the base
    # class) pending a reference-based definition; "higher is better" is not
    # meaningful without a noiseless reference value.


class QATOLE(Benchmark):
    def _build_circuit(self) -> tuple[QuantumCircuit, str, list[int]]:
        qasm_source, circuit_id, observable_qubits = _load_qasm_source(self.params)
        circuit = _build_ole_circuit(qasm_source, observable_qubits)
        return circuit, circuit_id, observable_qubits

    def dispatch_handler(self, device: "QuantumDevice") -> QATOLEData:
        circuit, circuit_id, observable_qubits = self._build_circuit()
        num_gates = sum(
            1 for instr in circuit.data
            if instr.operation.name not in ("barrier", "measure")
        )
        return QATOLEData.from_quantum_job(
            device.run(circuit, shots=self.params.shots),
            observable_qubits=observable_qubits,
            shots=self.params.shots,
            circuit_id=circuit_id,
            num_qubits=circuit.num_qubits,
            num_gates=num_gates,
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
            observable_value=BenchmarkScore(value=value, uncertainty=uncertainty),
            circuit_id=job_data.circuit_id,
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        circuit, _, _ = self._build_circuit()
        return [CircuitBatch(circuits=[circuit], shots=self.params.shots)]
