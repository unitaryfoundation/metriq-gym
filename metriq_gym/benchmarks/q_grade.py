"""Q-Grade benchmark implementation.

Summary:
    Implements the many-body quantum coherence test from Teng et al.
    (arXiv:2503.12573). For each ring size in ``ring_sizes`` the benchmark
    submits two Trotterised circuits that encode a tight-binding "particle on
    a ring" in an Ising-chain with twisted boundary conditions: one with a
    vison (magnetic pi flux) through the ring and one without. Anyonic
    statistics cause the vison case to destructively interfere the particle's
    arrival at the diametrically opposite site; decoherence gradually lifts
    that blockade. The contrast

        R_gamma(L) = (<n_{L/2}>^{nov}_shots - <n_{L/2}>^{v}_shots)
                     / (<n_{L/2}>^{nov}_trot  - <n_{L/2}>^{v}_trot)

    is normalised by the noiseless-Trotter values of the *same* circuit so
    that Trotter error cancels and R_gamma = 1 on a perfect device.

Result interpretation:
    Polling returns ``QGradeResult`` with:
        - ``q_grade``: largest ring size in ``ring_sizes`` for which R_gamma
          meets the configured threshold (default 0.2). Zero if none qualify.
        - ``r_gamma``: BenchmarkScore at the reported ring size, with binomial
          uncertainty (SM Note F, Eq. F3).
    The headline ``score`` exposed via ``compute_score`` is the Q-grade.

References:
    - [Teng, Scarlatella, Zhou, Rahmani, Chamon, Castelnovo,
      "Standardized test of many-body coherence in gate-based quantum
      platforms", arXiv:2503.12573 (2025)](https://arxiv.org/abs/2503.12573).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

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


# -------- Paper parameters (SM Note D) --------
# Closed-form extrapolations, valid for even L >= 4:
#   t_max(L) = 5 + 2.75 L   (linear fit to the paper's tabulated first-peak
#                            times; agrees to within rounding for L = 4..22)
#   N_opt(L) = L + 2        (empirical minimum for avg Trotter err <= 0.15)

def _validate_L(L: int) -> None:
    if not isinstance(L, int):
        raise TypeError(f"L must be an int, got {type(L).__name__}")
    if L < 4 or L % 2 != 0:
        raise ValueError(f"L must be an even integer >= 4, got L={L}")


def t_max(L: int) -> float:
    _validate_L(L)
    return 5.0 + 2.75 * L


def n_opt(L: int) -> int:
    _validate_L(L)
    return L + 2


# -------- Circuit construction --------

def _init_state_prep(qc: QuantumCircuit, n: int, vison: bool) -> None:
    """GHZ-style preparation of the single-spinon initial state.
    If ``vison`` is True, precede the Hadamard with an X to introduce a pi
    flux (vison) through the ring."""
    if vison:
        qc.x(0)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)


def _rzz(qc: QuantumCircuit, angle: float, i: int, j: int) -> None:
    """exp(-i * angle/2 * Z_i Z_j) compiled as CNOT - Rz - CNOT."""
    qc.cx(i, j)
    qc.rz(angle, j)
    qc.cx(i, j)


def _interaction_layer(qc: QuantumCircuit, n: int, angle: float) -> None:
    """One full ZZ layer on the ring: even bonds, then odd bonds, then the
    twisted bond (n-1, 0) with a MINUS sign (this implements J_0 = -1)."""
    for i in range(n // 2):
        _rzz(qc, angle, 2 * i, 2 * i + 1)
    for i in range(n // 2 - 1):
        _rzz(qc, angle, 2 * i + 1, 2 * i + 2)
    _rzz(qc, -angle, n - 1, 0)  # twisted bond


def _hopping_layer(qc: QuantumCircuit, n: int, angle: float) -> None:
    for i in range(n):
        qc.rx(angle, i)


def build_q_grade_circuit(
    L: int,
    vison: bool,
    J: float = 1.0,
    Gamma: float = 0.1,
    measure: bool = True,
) -> QuantumCircuit:
    """Build the L-site Trotterised circuit for the Q-grade benchmark."""
    _validate_L(L)
    t = t_max(L)
    n_steps = n_opt(L)
    dt = t / n_steps

    qc = QuantumCircuit(L, L) if measure else QuantumCircuit(L)
    _init_state_prep(qc, L, vison)
    for _ in range(n_steps):
        _interaction_layer(qc, L, 2 * J * dt)
        _hopping_layer(qc, L, 2 * Gamma * dt)
    if measure:
        qc.measure(range(L), range(L))
    return qc


# -------- Readout helpers --------

def _midpoint_from_indexed_probs(probs_by_idx, L: int) -> float:
    """<n_{L/2}> = Prob(domain wall across qubits L/2 - 1 and L/2).

    Qiskit convention: bits[q] = (idx >> q) & 1, qubit 0 is the least
    significant bit.
    """
    site = L // 2
    a, b = site - 1, site
    p = 0.0
    for idx, pr in probs_by_idx:
        if ((idx >> a) & 1) != ((idx >> b) & 1):
            p += pr
    return float(p)


def _midpoint_from_counts(counts: dict, L: int) -> float:
    shots = sum(counts.values())
    if shots == 0:
        return 0.0
    pairs = ((int(bs.replace(" ", ""), 2), c / shots) for bs, c in counts.items())
    return _midpoint_from_indexed_probs(pairs, L)


def _midpoint_noiseless_trotter(L: int, vison: bool, J: float, Gamma: float) -> float:
    """Expectation from the same Trotter circuit run as a pure statevector.
    Used as the R_gamma denominator so Trotter error cancels out."""
    qc = build_q_grade_circuit(L, vison, J, Gamma, measure=False)
    psi = Statevector(qc).data
    probs = ((idx, abs(psi[idx]) ** 2) for idx in range(2 ** L))
    return _midpoint_from_indexed_probs(probs, L)


# -------- Benchmark plumbing --------

class QGradeResult(BenchmarkResult):
    q_grade: int
    r_gamma: BenchmarkScore

    def compute_score(self) -> BenchmarkScore:
        # Expose Q-grade as headline. uncertainty=None (it is a thresholded
        # integer, not a statistical mean).
        return BenchmarkScore(value=float(self.q_grade), uncertainty=None)


@dataclass
class QGradeData(BenchmarkData):
    ring_sizes: list[int] = field(default_factory=list)
    shots: int = 1000
    threshold: float = 0.2
    # Noiseless-Trotter reference values, one entry per ring size (submission order).
    ref_vison: list[float] = field(default_factory=list)
    ref_novison: list[float] = field(default_factory=list)


class QGrade(Benchmark[QGradeData, QGradeResult]):
    def _build_circuits(self, device: "QuantumDevice") -> tuple[
        list[QuantumCircuit], list[float], list[float]
    ]:
        """Shared circuit construction: two circuits (vison, no-vison) per
        ring size, plus the matching noiseless-Trotter reference values."""
        ring_sizes: list[int] = list(self.params.ring_sizes)
        j_coupling: float = float(getattr(self.params, "j_coupling", 1.0))
        gamma_hopping: float = float(getattr(self.params, "gamma_hopping", 0.1))

        circuits: list[QuantumCircuit] = []
        ref_v: list[float] = []
        ref_nv: list[float] = []
        for L in ring_sizes:
            _validate_L(L)
            circuits.append(
                build_q_grade_circuit(L, vison=True,  J=j_coupling, Gamma=gamma_hopping)
            )
            circuits.append(
                build_q_grade_circuit(L, vison=False, J=j_coupling, Gamma=gamma_hopping)
            )
            ref_v.append(_midpoint_noiseless_trotter(L, True,  j_coupling, gamma_hopping))
            ref_nv.append(_midpoint_noiseless_trotter(L, False, j_coupling, gamma_hopping))

        return circuits, ref_v, ref_nv

    def dispatch_handler(self, device: "QuantumDevice") -> QGradeData:
        circuits, ref_v, ref_nv = self._build_circuits(device)
        return QGradeData.from_quantum_job(
            quantum_job=device.run(circuits, shots=self.params.shots),
            ring_sizes=list(self.params.ring_sizes),
            shots=self.params.shots,
            threshold=float(getattr(self.params, "threshold", 0.2)),
            ref_vison=ref_v,
            ref_novison=ref_nv,
        )

    def poll_handler(
        self,
        job_data: QGradeData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> QGradeResult:
        counts_list = flatten_counts(result_data)
        expected = 2 * len(job_data.ring_sizes)
        if len(counts_list) != expected:
            raise RuntimeError(
                f"Q-Grade: expected {expected} result batches (2 per ring size), "
                f"got {len(counts_list)}."
            )

        # R_gamma(L) for every ring size, plus bookkeeping for the headline score.
        r_by_L: dict[int, float] = {}
        n_v_by_L: dict[int, float] = {}
        n_nv_by_L: dict[int, float] = {}
        for k, L in enumerate(job_data.ring_sizes):
            n_v = _midpoint_from_counts(counts_list[2 * k], L)
            n_nv = _midpoint_from_counts(counts_list[2 * k + 1], L)
            denom = job_data.ref_novison[k] - job_data.ref_vison[k]
            r_by_L[L] = (n_nv - n_v) / denom if denom != 0 else 0.0
            n_v_by_L[L] = n_v
            n_nv_by_L[L] = n_nv

        # Q-grade: largest ring size clearing threshold. If nothing qualifies,
        # the headline is 0 but we still report R_gamma at the smallest ring
        # size tested so the user sees a non-trivial number.
        qualifying = [L for L, r in r_by_L.items() if r >= job_data.threshold]
        if qualifying:
            reported_L = max(qualifying)
            q_grade_value = reported_L
        else:
            reported_L = min(job_data.ring_sizes)
            q_grade_value = 0

        k_rep = job_data.ring_sizes.index(reported_L)
        denom = job_data.ref_novison[k_rep] - job_data.ref_vison[k_rep]
        shots = job_data.shots
        n_v = n_v_by_L[reported_L]
        n_nv = n_nv_by_L[reported_L]
        if shots > 0 and denom != 0:
            var = (
                (1 - n_v) * n_v + (1 - n_nv) * n_nv
            ) / (shots * denom ** 2)
            unc = float(np.sqrt(max(var, 0.0)))
        else:
            unc = None

        return QGradeResult(
            q_grade=q_grade_value,
            r_gamma=BenchmarkScore(value=float(r_by_L[reported_L]), uncertainty=unc),
        )

    def estimate_resources_handler(
        self, device: "QuantumDevice"
    ) -> list[CircuitBatch]:
        circuits, _, _ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
