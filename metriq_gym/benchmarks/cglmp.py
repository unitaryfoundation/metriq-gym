"""CGLMP-d benchmark implementation.

Summary:
    Two-party benchmark for the Collins-Gisin-Linden-Massar-Popescu Bell
    inequality I_d on d-level systems (qudits). Each party measures one
    of two settings with d possible outcomes. The benchmark prepares the
    maximally entangled qudit state |Phi_d> = (1/sqrt(d)) sum_j |jj>
    across a chosen set of qubits (using a binary encoding for d a power
    of 2), applies the Acin et al. optimal Fourier-basis measurements,
    and reports the empirical CGLMP value.

Supported dimensions:
    d = 2: reduces to CHSH on a single edge. Two qubits required.
    d = 4: nontrivial qudit case using 2 qubits per party (4 qubits
        total). Uses the standard 2-qubit QFT for the Fourier transform.
    d = 3 is not supported (needs leakage suppression on 2 qubits to
    avoid amplitude leaking into |11>, the 4th basis state).

Bounds:
    - Classical (separable / LHV): I_d <= 2 for all d.
    - Quantum maximum on the maximally entangled state (tabulated from
      Acin et al., PRA 65, 052325): 2*sqrt(2) for d=2, ~2.896 for d=4.

Result interpretation:
    Polling returns CGLMPResult with:
        - expectation_value: estimated I_d in [-quantum_bound,
          quantum_bound] under ideal execution.
        - achievement_ratio: (expectation_value - 2) / (quantum_bound - 2).
          Positive values indicate a CGLMP violation.
        - violated: bool, true iff expectation_value > 2 (the classical
          bound).

References:
    - Collins, Gisin, Linden, Massar, Popescu, "Bell inequalities for
      arbitrarily high-dimensional systems", *Phys. Rev. Lett.* 88,
      040404 (2002).
    - Acin, Andrianov, Costa, Jane, Latorre, Tarrach, "Quantum
      nonlocality in two three-level systems", *Phys. Rev. A* 65,
      052325 (2002).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qiskit import QuantumCircuit
from qiskit.synthesis import synth_qft_full

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.resource_estimation import CircuitBatch


SUPPORTED_D: tuple[int, ...] = (2, 4)

# Tabulated quantum maxima for CGLMP-d on the maximally entangled state
# with Acin et al. optimal Fourier-basis measurements.
_TABULATED_QUANTUM_BOUNDS: dict[int, float] = {
    2: 2.0 * math.sqrt(2.0),
    4: 2.896194,
}

CLASSICAL_BOUND = 2.0


def quantum_bound(d: int) -> float:
    """Tabulated quantum maximum of I_d on the maximally entangled state."""
    if d not in _TABULATED_QUANTUM_BOUNDS:
        raise NotImplementedError(
            f"Quantum bound for CGLMP-{d} not tabulated; supported d: {sorted(_TABULATED_QUANTUM_BOUNDS)}"
        )
    return _TABULATED_QUANTUM_BOUNDS[d]


def _check_d(d: int) -> None:
    if d not in SUPPORTED_D:
        raise NotImplementedError(
            f"CGLMP qubit-encoded circuits implemented only for d in {SUPPORTED_D}; got d={d}."
        )


def num_qubits_per_party(d: int) -> int:
    """Number of qubits encoding a single qudit of dimension d (binary)."""
    _check_d(d)
    return int(math.log2(d))


def optimal_alice_phases() -> tuple[float, float]:
    """Acin et al. optimal Alice phases: (alpha_1, alpha_2) = (0, 1/2)."""
    return (0.0, 0.5)


def optimal_bob_phases() -> tuple[float, float]:
    """Acin et al. optimal Bob phases: (beta_1, beta_2) = (+1/4, -1/4)."""
    return (0.25, -0.25)


def _phase_diag_circuit(num_qubits: int, alpha: float, d: int) -> QuantumCircuit:
    """Append D_alpha = diag(exp(i 2 pi j alpha / d)) on num_qubits qubits.

    Diagonal-by-component: the j-th component phase factorises as the
    product of single-qubit phases ``exp(i 2 pi alpha 2^k / d)`` on qubit
    k (LSB convention; qubit 0 is the least significant bit of j).
    """
    qc = QuantumCircuit(num_qubits, name=f"D({alpha:+.3g})")
    if alpha == 0.0:
        return qc
    for k in range(num_qubits):
        qc.p(2 * math.pi * alpha * (2**k) / d, k)
    return qc


def _alice_basis_change(num_qubits: int, alpha: float, d: int) -> QuantumCircuit:
    """U_A = F_d^dagger D_{-alpha}: maps Alice's measurement basis to comp basis."""
    qc = QuantumCircuit(num_qubits, name=f"U_A(alpha={alpha})")
    qc.compose(_phase_diag_circuit(num_qubits, -alpha, d), inplace=True)
    qc.compose(synth_qft_full(num_qubits, do_swaps=True).inverse(), inplace=True)
    return qc


def _bob_basis_change(num_qubits: int, beta: float, d: int) -> QuantumCircuit:
    """U_B = F_d D_{-beta}: maps Bob's measurement basis to comp basis."""
    qc = QuantumCircuit(num_qubits, name=f"U_B(beta={beta})")
    qc.compose(_phase_diag_circuit(num_qubits, -beta, d), inplace=True)
    qc.compose(synth_qft_full(num_qubits, do_swaps=True), inplace=True)
    return qc


def prepare_max_entangled(
    d: int, alice_qubits: list[int], bob_qubits: list[int], num_qubits: int
) -> QuantumCircuit:
    """Prepare |Phi_d> = (1/sqrt(d)) sum_j |j>|j> across Alice and Bob qubits.

    Under binary encoding, the maximally entangled qudit state factorises
    into q parallel Bell pairs, each pairing Alice's k-th qubit with
    Bob's k-th qubit.
    """
    q = num_qubits_per_party(d)
    if len(alice_qubits) != q or len(bob_qubits) != q:
        raise ValueError(f"alice/bob_qubits must each have length {q} for d={d}")
    qc = QuantumCircuit(num_qubits, name=f"|Phi_{d}>")
    for k in range(q):
        qc.h(alice_qubits[k])
        qc.cx(alice_qubits[k], bob_qubits[k])
    return qc


def build_cglmp_circuits(
    d: int, alice_qubits: list[int], bob_qubits: list[int], num_qubits: int
) -> list[QuantumCircuit]:
    """Build the four CGLMP-d measurement circuits, one per (x, y) in {1,2}^2.

    Each circuit prepares the maximally entangled state and applies the
    Acin et al. optimal Fourier-basis measurement for the chosen
    settings. Classical bits 0..q-1 hold Alice's outcome bits
    (LSB-first), bits q..2q-1 hold Bob's. The order of returned circuits
    is (x=1,y=1), (x=1,y=2), (x=2,y=1), (x=2,y=2).
    """
    q = num_qubits_per_party(d)
    alpha = optimal_alice_phases()
    beta = optimal_bob_phases()
    circuits: list[QuantumCircuit] = []
    for x in (1, 2):
        for y in (1, 2):
            qc = QuantumCircuit(num_qubits, 2 * q, name=f"CGLMP_{d}_x{x}_y{y}")
            qc.compose(
                prepare_max_entangled(d, alice_qubits, bob_qubits, num_qubits),
                inplace=True,
            )
            qc.compose(
                _alice_basis_change(q, alpha[x - 1], d), qubits=alice_qubits, inplace=True
            )
            qc.compose(
                _bob_basis_change(q, beta[y - 1], d), qubits=bob_qubits, inplace=True
            )
            qc.measure(alice_qubits, list(range(q)))
            qc.measure(bob_qubits, list(range(q, 2 * q)))
            circuits.append(qc)
    return circuits


def counts_to_probs(
    counts: dict[str, int], d: int, x: int, y: int
) -> dict[tuple[int, int, int, int], float]:
    """Convert raw counts into the joint probability table P(A_x = a, B_y = b).

    Returned dict is keyed by (x, y, a, b). Qiskit's right-to-left bit
    convention is used: cleaned[-1] is classical bit 0.
    """
    q = num_qubits_per_party(d)
    total = sum(counts.values())
    out: dict[tuple[int, int, int, int], float] = {}
    if total == 0:
        return out
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        a_bits = [int(cleaned[-1 - i]) for i in range(q)]
        b_bits = [int(cleaned[-1 - (q + i)]) for i in range(q)]
        a = sum(bit << i for i, bit in enumerate(a_bits))
        b = sum(bit << i for i, bit in enumerate(b_bits))
        out[(x, y, a, b)] = out.get((x, y, a, b), 0.0) + count / total
    return out


def cglmp_score(probs: dict[tuple[int, int, int, int], float], d: int) -> float:
    """Evaluate I_d on a joint probability table.

    Sum is over k = 0..floor(d/2)-1 with weights f(k) = 1 - 2k/(d-1).
    Each weight multiplies (plus_term - minus_term), where each term is
    a sum of four joint probabilities indexed by mod-d residue
    conditions on (a - b) and (b - a).
    """
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}")

    def joint(x: int, y: int, predicate) -> float:
        return sum(
            probs.get((x, y, a, b), 0.0)
            for a in range(d)
            for b in range(d)
            if predicate(a, b)
        )

    score = 0.0
    for k in range(d // 2):
        weight = 1.0 - 2.0 * k / (d - 1)
        plus = (
            joint(1, 1, lambda a, b, k=k: (a - b) % d == k % d)
            + joint(2, 1, lambda a, b, k=k: (b - a) % d == (k + 1) % d)
            + joint(2, 2, lambda a, b, k=k: (a - b) % d == k % d)
            + joint(1, 2, lambda a, b, k=k: (b - a) % d == k % d)
        )
        minus = (
            joint(1, 1, lambda a, b, k=k: (a - b) % d == (d - k - 1) % d)
            + joint(2, 1, lambda a, b, k=k: (b - a) % d == (d - k) % d)
            + joint(2, 2, lambda a, b, k=k: (a - b) % d == (d - k - 1) % d)
            + joint(1, 2, lambda a, b, k=k: (b - a) % d == (d - k - 1) % d)
        )
        score += weight * (plus - minus)
    return score


def ideal_quantum_probs(d: int) -> dict[tuple[int, int, int, int], float]:
    """Analytic joint probabilities for the maximally entangled state.

    Useful for testing cglmp_score without simulating circuits.
    """
    alpha = optimal_alice_phases()
    beta = optimal_bob_phases()

    def prob(x: int, y: int, a: int, b: int) -> float:
        amp = 0.0 + 0.0j
        for j in range(d):
            amp += np.exp(
                -1j * 2 * np.pi * j * ((a + alpha[x - 1]) - (b - beta[y - 1])) / d
            )
        amp /= d ** 1.5
        return float(np.abs(amp) ** 2)

    out: dict[tuple[int, int, int, int], float] = {}
    for x in (1, 2):
        for y in (1, 2):
            for a in range(d):
                for b in range(d):
                    out[(x, y, a, b)] = prob(x, y, a, b)
    return out


class CGLMPResult(BenchmarkResult):
    expectation_value: float
    achievement_ratio: float
    violated: bool

    def _iter_metric_items(self):
        yield "expectation_value", float(self.expectation_value), None
        yield "achievement_ratio", float(self.achievement_ratio), None
        yield "violated", float(self.violated), None

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=float(self.expectation_value))


@dataclass
class CGLMPData(BenchmarkData):
    shots: int = 0
    d: int = 2
    qubits: list[int] = field(default_factory=lambda: [0, 1])
    num_qubits: int = 0


class CGLMP(Benchmark):
    """Benchmark class for the CGLMP-d Bell inequality."""

    def _partition_qubits(self) -> tuple[list[int], list[int]]:
        d = self.params.d
        if d not in SUPPORTED_D:
            raise NotImplementedError(
                f"CGLMP supports d in {SUPPORTED_D}; got d={d}."
            )
        q = num_qubits_per_party(d)
        qs = list(self.params.qubits)
        if len(qs) != 2 * q:
            raise ValueError(
                f"CGLMP-d={d} requires {2 * q} qubits (q={q} per party); got {qs}"
            )
        if len(set(qs)) != 2 * q:
            raise ValueError(f"CGLMP requires distinct qubits; got {qs}")
        return qs[:q], qs[q:]

    def dispatch_handler(self, device: QuantumDevice) -> CGLMPData:
        shots = self.params.shots
        d = self.params.d
        alice_qubits, bob_qubits = self._partition_qubits()
        num_qubits = device.num_qubits
        circuits = build_cglmp_circuits(d, alice_qubits, bob_qubits, num_qubits)
        # Decompose so backends that don't understand the opaque QFT
        # instruction get a basis-gate form.
        circuits = [c.decompose(reps=4) for c in circuits]
        return CGLMPData.from_quantum_job(
            device.run(circuits, shots=shots),
            shots=shots,
            d=d,
            qubits=alice_qubits + bob_qubits,
            num_qubits=num_qubits,
        )

    def poll_handler(
        self,
        job_data: CGLMPData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> CGLMPResult:
        counts_list = flatten_counts(result_data)
        if len(counts_list) != 4:
            raise ValueError(f"expected 4 count dicts, got {len(counts_list)}")
        d = job_data.d
        probs: dict[tuple[int, int, int, int], float] = {}
        idx = 0
        for x in (1, 2):
            for y in (1, 2):
                probs.update(counts_to_probs(counts_list[idx], d, x, y))
                idx += 1
        score = cglmp_score(probs, d)
        qb = quantum_bound(d)
        gap = qb - CLASSICAL_BOUND
        achievement_ratio = (score - CLASSICAL_BOUND) / gap if gap > 0 else 0.0
        return CGLMPResult(
            expectation_value=score,
            achievement_ratio=achievement_ratio,
            violated=score > CLASSICAL_BOUND,
        )

    def estimate_resources_handler(
        self,
        device: QuantumDevice,
    ) -> list[CircuitBatch]:
        d = self.params.d
        alice_qubits, bob_qubits = self._partition_qubits()
        circuits = build_cglmp_circuits(d, alice_qubits, bob_qubits, device.num_qubits)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
