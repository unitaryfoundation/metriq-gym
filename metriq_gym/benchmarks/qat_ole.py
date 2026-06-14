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
    - noiseless_reference: the exact OLE value over the same sampled initial
      states, computed by statevector simulation when the circuit is small
      enough (at most 20 qubits). None for the named QAT circuits, which sit
      beyond classical simulation.
    - score: observable_value / noiseless_reference when a reference is
      available (1.0 means noiseless; decoherence drives it toward 0).
      Unset when no classical reference exists.

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
from typing import TYPE_CHECKING, NamedTuple

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit import qasm3
from qiskit.quantum_info import SparsePauliOp, Statevector
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

# Largest circuit for which a noiseless statevector reference is computed.
_MAX_REFERENCE_QUBITS = 20

_CIRCUIT_FILENAMES: dict[str, str] = {
    "49Q_L3": "49Q_OLE_circuit_L_3_b_0.25_delta0.15.qasm",
    "49Q_L6": "49Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm",
    "70Q_L6": "70Q_OLE_circuit_L_6_b_0.25_delta0.15.qasm",
}


def _fetch_qasm(circuit_name: str) -> str:
    if circuit_name not in _CIRCUIT_FILENAMES:
        raise ValueError(
            f"Unknown circuit {circuit_name!r}. "
            f"Supported circuits: {', '.join(sorted(_CIRCUIT_FILENAMES))}"
        )
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
        # Copy so callers can't mutate the module-level list.
        return _fetch_qasm(circuit_name), circuit_name, list(_OBSERVABLE_QUBITS)
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
    # qasm3.loads() requires the optional qiskit-qasm3-import package,
    # declared explicitly via the qiskit[qasm3-import] extra in pyproject.toml.
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


def _active_qubits(circuit: QuantumCircuit) -> set[int]:
    """Qubit indices touched by at least one non-barrier operation."""
    active: set[int] = set()
    for instr in circuit.data:
        if instr.operation.name == "barrier":
            continue
        for q in instr.qubits:
            active.add(circuit.find_bit(q).index)
    return active


def _sample_initial_states(
    num_qubits: int,
    num_states: int,
    rng: random.Random,
    active_qubits: set[int] | None = None,
) -> list[str]:
    """Sample computational-basis initial states uniformly at random.

    Each state is a bitstring where index i is the value of qubit i. Only
    qubits in active_qubits get random bits; the rest stay '0'. The named QAT
    circuits declare 156 physical qubits but only act on the 49/70-qubit
    problem lattice, and random X preparations on idle padding qubits would
    add hardware noise and crosstalk without affecting the echo.
    """
    if active_qubits is None:
        active_qubits = set(range(num_qubits))
    return [
        "".join(rng.choice("01") if q in active_qubits else "0" for q in range(num_qubits))
        for _ in range(num_states)
    ]


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


def _noiseless_reference(
    base: QuantumCircuit, initial_states: list[str], observable_qubits: list[int]
) -> float | None:
    """Exact OLE value over the same sampled initial states, by statevector.

    Gives the device-independent target the measured estimate is compared
    against in compute_score. Only available when the circuit is small enough
    to simulate classically; returns None above _MAX_REFERENCE_QUBITS (the
    named QAT circuits declare 156 qubits, where no classical reference
    exists; that regime is the point of the QAT).
    """
    if base.num_qubits > _MAX_REFERENCE_QUBITS:
        return None
    observable = SparsePauliOp.from_sparse_list(
        [("Z" * len(observable_qubits), observable_qubits, 1.0)],
        num_qubits=base.num_qubits,
    )
    total = 0.0
    for state in initial_states:
        qc = QuantumCircuit(base.num_qubits)
        for q, bit in enumerate(state):
            if bit == "1":
                qc.x(q)
        qc.compose(base, inplace=True)
        expectation = Statevector(qc).expectation_value(observable).real
        total += _initial_state_sign(state, observable_qubits) * expectation
    return total / len(initial_states)


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

    Returns (value, uncertainty). For multiple initial states the uncertainty
    is the standard error of the mean of the per-state estimates. Their spread
    already includes finite-shot noise (each estimate is itself shot-noisy),
    so no separate shot term is added — that would double count. With a single
    initial state there is no spread to measure, so the propagated shot noise
    of that one estimate is used instead.
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
    if n > 1:
        sample_var = sum((e - value) ** 2 for e in estimates) / (n - 1)
        uncertainty = sqrt(sample_var / n)
    else:
        uncertainty = sqrt(shot_variances[0])
    return value, uncertainty


@dataclass
class QATOLEData(BenchmarkData):
    observable_qubits: list[int]
    initial_states: list[str]
    seed: int | None
    shots: int
    circuit_id: str
    num_qubits: int
    num_gates: int
    noiseless_reference: float | None = None


class QATOLEResult(BenchmarkResult):
    observable_value: BenchmarkScore
    noiseless_reference: float | None = None

    def compute_score(self) -> BenchmarkScore | None:
        """Score = measured OLE divided by the noiseless reference.

        1.0 means the device reproduced the ideal echo; decoherence drives the
        ratio toward 0. Only defined when a classical reference is available
        (circuits small enough for statevector simulation). For the named QAT
        circuits no reference is classically computable, so score stays unset.
        """
        if self.noiseless_reference is None or abs(self.noiseless_reference) < 1e-9:
            return None
        uncertainty = self.observable_value.uncertainty
        return BenchmarkScore(
            value=self.observable_value.value / self.noiseless_reference,
            uncertainty=(
                abs(uncertainty / self.noiseless_reference) if uncertainty is not None else None
            ),
        )


class _BuiltCircuits(NamedTuple):
    """Everything _build_circuits produces for dispatch.

    Attributes:
        circuits: One measured OLE circuit per sampled initial state.
        circuit_id: Named-circuit name, or the basename of qasm_path.
        observable_qubits: Indices of the Z-product observable qubits.
        initial_states: The sampled computational-basis states, as bitstrings.
        num_gates: Gate count of the base circuit (barriers/measures excluded).
        noiseless_reference: Exact OLE over the sampled states via statevector,
            or None when the circuit is too large to simulate classically.
    """

    circuits: list[QuantumCircuit]
    circuit_id: str
    observable_qubits: list[int]
    initial_states: list[str]
    num_gates: int
    noiseless_reference: float | None


class QATOLE(Benchmark):
    def _build_circuits(self) -> _BuiltCircuits:
        """Load the base circuit, sample initial states, and build the run set.

        QASM loading itself lives in _load_qasm_source/_parse_and_validate;
        this method layers the OLE-specific parts on top (initial-state
        sampling, preparation/measurement layers, noiseless reference).
        """
        qasm_source, circuit_id, observable_qubits = _load_qasm_source(self.params)
        base = _parse_and_validate(qasm_source, observable_qubits)
        num_initial_states = getattr(self.params, "num_initial_states", _DEFAULT_NUM_INITIAL_STATES)
        seed = getattr(self.params, "seed", None)
        rng = random.Random(seed)
        # Restrict sampling to qubits the circuit actually acts on: the named
        # QAT circuits pad to 156 declared qubits. Observable qubits are always
        # included so the parity sign factor stays well-defined.
        active = _active_qubits(base) | set(observable_qubits)
        initial_states = _sample_initial_states(base.num_qubits, num_initial_states, rng, active)
        circuits = [_prepare_circuit(base, state, observable_qubits) for state in initial_states]
        num_gates = sum(
            1 for instr in base.data if instr.operation.name not in ("barrier", "measure")
        )
        return _BuiltCircuits(
            circuits=circuits,
            circuit_id=circuit_id,
            observable_qubits=observable_qubits,
            initial_states=initial_states,
            num_gates=num_gates,
            noiseless_reference=_noiseless_reference(base, initial_states, observable_qubits),
        )

    def dispatch_handler(self, device: "QuantumDevice") -> QATOLEData:
        built = self._build_circuits()
        return QATOLEData.from_quantum_job(
            device.run(built.circuits, shots=self.params.shots),
            observable_qubits=built.observable_qubits,
            initial_states=built.initial_states,
            seed=getattr(self.params, "seed", None),
            shots=self.params.shots,
            circuit_id=built.circuit_id,
            num_qubits=built.circuits[0].num_qubits,
            num_gates=built.num_gates,
            noiseless_reference=built.noiseless_reference,
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
            noiseless_reference=job_data.noiseless_reference,
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        built = self._build_circuits()
        return [CircuitBatch(circuits=built.circuits, shots=self.params.shots)]
