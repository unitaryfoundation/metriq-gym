"""QAT Operator Loschmidt Echo (OLE) benchmark implementation.

Summary:
    Loads an OLE circuit from the Quantum Advantage Tracker (by name) or from
    a local OpenQASM 3.0 file (via qasm_path) and estimates the operator
    Loschmidt echo of the Pauli-Z product observable O = Z_{q_0} ... Z_{q_k}.

    Following the QAT protocol, the estimator averages over randomly sampled
    computational-basis initial states. For each sampled state |x⟩ the circuit
    is run with an X-gate preparation layer, the observable qubits are
    measured, and the measured parity expectation m_x = ⟨Z_{q_0} ... Z_{q_k}⟩
    is weighted by the initial-state parity factor
    σ_z(x) = ⟨x|Z_{q_0} ... Z_{q_k}|x⟩ = ±1:

        OLE ≈ (1 / N_init) Σ_x σ_z(x) · m_x

Result interpretation:
    - observable_value: the OLE estimate averaged over the sampled initial
      states. Its uncertainty combines per-state shot noise and the sampling
      spread across initial states.
    - circuit_id: name of the named circuit or basename of qasm_path.
    Score is left unset pending a reference-based definition.

Local simulator usage:
    Create an example config referencing a small fixture circuit:

        {
          "benchmark_name": "QAT OLE",
          "qasm_path": "metriq_gym/schemas/examples/qat_ole_small.qasm",
          "observable_qubits": [0, 1, 2],
          "num_initial_states": 5,
          "shots": 100,
          "seed": 42
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
          "num_initial_states": 10,
          "shots": 1000
        }

    The named circuits are pre-compiled to 156 physical qubits and require a
    large-scale device such as IBM Eagle or IBM Heron. Note that the device
    executes num_initial_states circuits, each with the configured shots.

References:
    - QAT OLE circuits: https://github.com/quantum-advantage-tracker/
      quantum-advantage-tracker.github.io/tree/main/data/observable-estimations/
      circuit-models/operator_loschmidt_echo
    - Algorithmiq model description:
      https://algorithmiq.fi/wp-content/uploads/2025/11/model-information-flow-complex-material-document.pdf
"""

from __future__ import annotations

import os
import random
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

_DEFAULT_NUM_INITIAL_STATES = 10

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


def _parse_and_validate(qasm_source: str, observable_qubits: list[int]) -> QuantumCircuit:
    """Parse a QASM 3.0 source string and validate it against the observable.

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
            f"observable_qubits {out_of_range} are out of range for a {num_qubits}-qubit circuit"
        )
    return qc


def _sample_initial_states(num_qubits: int, num_states: int, rng: random.Random) -> list[str]:
    """Sample computational-basis initial states uniformly at random.

    Each state is a bitstring where index i is the value of qubit i.
    """
    return ["".join(rng.choice("01") for _ in range(num_qubits)) for _ in range(num_states)]


def _initial_state_sign(initial_state: str, observable_qubits: list[int]) -> int:
    """Return σ_z(x) = ⟨x|Z_{q_0} ... Z_{q_k}|x⟩ = ±1 for a basis state x."""
    parity = sum(int(initial_state[q]) for q in observable_qubits) % 2
    return -1 if parity else 1


def _prepare_circuit(
    base: QuantumCircuit, initial_state: str, observable_qubits: list[int]
) -> QuantumCircuit:
    """Prepend X-gate state preparation and append observable measurements."""
    qc = QuantumCircuit(base.num_qubits)
    for q, bit in enumerate(initial_state):
        if bit == "1":
            qc.x(q)
    qc.compose(base, inplace=True)
    cr = ClassicalRegister(len(observable_qubits), "c")
    qc.add_register(cr)
    for i, qubit_idx in enumerate(observable_qubits):
        qc.measure(qubit_idx, cr[i])
    return qc


def _build_ole_circuits(
    qasm_source: str, observable_qubits: list[int], initial_states: list[str]
) -> list[QuantumCircuit]:
    """Build one measured OLE circuit per sampled initial state."""
    base = _parse_and_validate(qasm_source, observable_qubits)
    return [_prepare_circuit(base, state, observable_qubits) for state in initial_states]


def _pauli_z_product_expectation(counts: dict[str, int]) -> tuple[float, float]:
    """Compute ⟨Z_{q0} ... Z_{qk}⟩ from measurement-count bitstrings.

    Even-parity outcomes contribute +1, odd-parity outcomes contribute −1.
    Returns (expectation, uncertainty) where uncertainty is the propagated
    binomial standard deviation (2σ from the parity fraction).
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


def _ole_estimate(
    counts_list: list[dict[str, int]],
    initial_states: list[str],
    observable_qubits: list[int],
) -> tuple[float, float]:
    """Average sign-weighted parity expectations over the sampled initial states.

    Returns (value, uncertainty). The uncertainty combines, in quadrature:
    - shot noise propagated from each per-state parity estimate, and
    - the sampling spread across initial states (standard error of the mean).
    """
    if len(counts_list) != len(initial_states):
        raise ValueError(
            f"Got {len(counts_list)} count dictionaries for {len(initial_states)} initial states"
        )
    estimates: list[float] = []
    shot_variances: list[float] = []
    for counts, state in zip(counts_list, initial_states):
        m, u = _pauli_z_product_expectation(counts)
        sign = _initial_state_sign(state, observable_qubits)
        estimates.append(sign * m)
        shot_variances.append(u * u)

    n = len(estimates)
    value = sum(estimates) / n
    shot_term_sq = sum(shot_variances) / (n * n)
    if n > 1:
        sample_var = sum((e - value) ** 2 for e in estimates) / (n - 1)
        sample_term_sq = sample_var / n
    else:
        sample_term_sq = 0.0
    return value, sqrt(shot_term_sq + sample_term_sq)


@dataclass
class QATOLEData(BenchmarkData):
    observable_qubits: list[int]
    initial_states: list[str]
    seed: int | None
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
    def _build_circuits(self) -> tuple[list[QuantumCircuit], str, list[int], list[str], int]:
        qasm_source, circuit_id, observable_qubits = _load_qasm_source(self.params)
        base = _parse_and_validate(qasm_source, observable_qubits)
        num_initial_states = getattr(self.params, "num_initial_states", _DEFAULT_NUM_INITIAL_STATES)
        seed = getattr(self.params, "seed", None)
        rng = random.Random(seed)
        initial_states = _sample_initial_states(base.num_qubits, num_initial_states, rng)
        circuits = [_prepare_circuit(base, state, observable_qubits) for state in initial_states]
        num_gates = sum(
            1 for instr in base.data if instr.operation.name not in ("barrier", "measure")
        )
        return circuits, circuit_id, observable_qubits, initial_states, num_gates

    def dispatch_handler(self, device: "QuantumDevice") -> QATOLEData:
        circuits, circuit_id, observable_qubits, initial_states, num_gates = self._build_circuits()
        return QATOLEData.from_quantum_job(
            device.run(circuits, shots=self.params.shots),
            observable_qubits=observable_qubits,
            initial_states=initial_states,
            seed=getattr(self.params, "seed", None),
            shots=self.params.shots,
            circuit_id=circuit_id,
            num_qubits=circuits[0].num_qubits,
            num_gates=num_gates,
        )

    def poll_handler(
        self,
        job_data: QATOLEData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> QATOLEResult:
        counts_list = flatten_counts(result_data)
        value, uncertainty = _ole_estimate(
            counts_list, job_data.initial_states, job_data.observable_qubits
        )
        return QATOLEResult(
            observable_value=BenchmarkScore(value=value, uncertainty=uncertainty),
            circuit_id=job_data.circuit_id,
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        circuits, _, _, _, _ = self._build_circuits()
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
