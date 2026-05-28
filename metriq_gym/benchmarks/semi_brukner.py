"""Semi-Brukner / Local Friendliness benchmark implementation.

Summary:
    Two-party benchmark evaluating the semi-Brukner inequality of
    Brukner (2018), one inequality in the family of Local Friendliness
    inequalities derived in Bong et al. (2020). The benchmark prepares a
    singlet on a chosen pair of qubits and measures four correlators
    corresponding to the terms of the semi-Brukner expression

        B = -A_1 (x) B_2 + A_1 (x) B_3 - A_3 (x) B_2 - A_3 (x) B_3,

    where A_x = cos(alpha_x) X + sin(alpha_x) Y on Alice's qubit and
    B_y = cos(gamma_y) X + sin(gamma_y) Y on Bob's qubit. The benchmark
    uses the angles that saturate the quantum bound 2*sqrt(2) on the
    singlet: alpha_1 = 0, alpha_3 = pi/2, gamma_2 = pi/4, gamma_3 = 3*pi/4.

Result interpretation:
    Polling returns SemiBruknerResult with:
        - expectation_value: estimated <B>, in [-2*sqrt(2), 2*sqrt(2)] under
          ideal execution.
        - achievement_ratio: (expectation_value - 2) / (2*sqrt(2) - 2).
          Positive values indicate a semi-Brukner violation, which under
          the assumptions of absoluteness of observed events and local
          agency rules out any Local Friendliness model.
        - lf_certified: bool, true iff expectation_value > 2 (the LF /
          fully separable bound).

References:
    - Brukner, "A no-go theorem for observer-independent facts",
      *Entropy* 20, 350 (2018).
    - Bong et al., "A strong no-go theorem on the Wigner's friend paradox",
      *Nature Phys.* 16, 1199 (2020).
    - Zeng, Labib, Russo, "Towards violations of Local Friendliness with
      quantum computers", arXiv:2409.15302.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit

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


# The four (Alice, Bob) setting pairs in
# B = -A_1 B_2 + A_1 B_3 - A_3 B_2 - A_3 B_3, paired with coefficients.
SEMI_BRUKNER_TERMS: list[tuple[str, str, float]] = [
    ("A1", "B2", -1.0),
    ("A1", "B3", +1.0),
    ("A3", "B2", -1.0),
    ("A3", "B3", -1.0),
]


def optimal_angles() -> dict[str, float]:
    """Angles (in radians) saturating <B> = 2*sqrt(2) on the singlet.

    Choice: alpha_1 = 0, alpha_3 = pi/2, gamma_2 = pi/4, gamma_3 = 3*pi/4.
    Verified by substitution into
    tr(B |psi-><psi-|) = cos(alpha_1 - gamma_2) - cos(alpha_1 - gamma_3)
    + cos(alpha_3 - gamma_2) + cos(alpha_3 - gamma_3),
    which evaluates to 4/sqrt(2) = 2*sqrt(2).
    """
    return {
        "A1": 0.0,
        "A3": math.pi / 2,
        "B2": math.pi / 4,
        "B3": 3 * math.pi / 4,
    }


def prepare_singlet(num_qubits: int, qubits: tuple[int, int]) -> QuantumCircuit:
    """Prepare the singlet (|01> - |10>)/sqrt(2) on the chosen qubit pair.

    Construction: starting from |00>, apply X then H on Alice, X on Bob,
    then CNOT(Alice, Bob). Direct verification:
        |00> -X_A-> |10> -H_A-> (|00> - |10>)/sqrt(2)
              -X_B-> (|01> - |11>)/sqrt(2)
              -CNOT_AB-> (|01> - |10>)/sqrt(2) = |psi->.
    """
    qa, qb = qubits
    qc = QuantumCircuit(num_qubits, name="singlet")
    qc.x(qa)
    qc.h(qa)
    qc.x(qb)
    qc.cx(qa, qb)
    return qc


def build_semi_brukner_circuits(
    qubits: tuple[int, int], num_qubits: int
) -> list[QuantumCircuit]:
    """Build the four semi-Brukner measurement circuits.

    Each circuit prepares the singlet on ``qubits``, applies the per-qubit
    basis change ``RZ(angle); H`` so the subsequent Z-basis measurement
    estimates the equatorial-Pauli correlator (up to an overall sign that
    drops out of the score by the parity ``<A(alpha) (x) B(gamma)>_{psi-}
    = -cos(alpha - gamma)``), then measures both qubits into classical
    bits 0 and 1. Returned in the order of ``SEMI_BRUKNER_TERMS``.
    """
    angles = optimal_angles()
    qa, qb = qubits
    circuits: list[QuantumCircuit] = []
    for a_label, b_label, _ in SEMI_BRUKNER_TERMS:
        qc = QuantumCircuit(num_qubits, 2)
        qc.compose(prepare_singlet(num_qubits, qubits), inplace=True)
        qc.rz(angles[a_label], qa)
        qc.h(qa)
        qc.rz(angles[b_label], qb)
        qc.h(qb)
        qc.measure(qa, 0)
        qc.measure(qb, 1)
        circuits.append(qc)
    return circuits


def correlator_from_counts(counts: dict[str, int]) -> float:
    """Estimate <A (x) B> = <(-1)^(a+b)> from a 2-qubit Z-basis count dict."""
    total = 0
    weighted = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[-2:]
        parity = sum(int(b) for b in bits) % 2
        weighted += count * (1 if parity == 0 else -1)
        total += count
    if total == 0:
        return 0.0
    return weighted / total


class SemiBruknerResult(BenchmarkResult):
    expectation_value: float
    achievement_ratio: float
    lf_certified: bool

    def _iter_metric_items(self):
        yield "expectation_value", float(self.expectation_value), None
        yield "achievement_ratio", float(self.achievement_ratio), None
        yield "lf_certified", float(self.lf_certified), None

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=float(self.expectation_value))


@dataclass
class SemiBruknerData(BenchmarkData):
    shots: int = 0
    qubits: list[int] = field(default_factory=lambda: [0, 1])
    num_qubits: int = 0


class SemiBrukner(Benchmark):
    """Benchmark class for the semi-Brukner / Local Friendliness inequality."""

    def _qubits(self) -> tuple[int, int]:
        qs = list(self.params.qubits)
        if len(qs) != 2:
            raise ValueError(f"semi-Brukner requires exactly 2 qubits, got {qs}")
        if qs[0] == qs[1]:
            raise ValueError(f"semi-Brukner requires two distinct qubits, got {qs}")
        return (qs[0], qs[1])

    def dispatch_handler(self, device: "QuantumDevice") -> SemiBruknerData:
        shots = self.params.shots
        qubits = self._qubits()
        num_qubits = device.num_qubits
        circuits = build_semi_brukner_circuits(qubits, num_qubits)
        return SemiBruknerData.from_quantum_job(
            device.run(circuits, shots=shots),
            shots=shots,
            qubits=list(qubits),
            num_qubits=num_qubits,
        )

    def poll_handler(
        self,
        job_data: SemiBruknerData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> SemiBruknerResult:
        counts = flatten_counts(result_data)
        if len(counts) != len(SEMI_BRUKNER_TERMS):
            raise ValueError(
                f"expected {len(SEMI_BRUKNER_TERMS)} count dicts, got {len(counts)}"
            )
        b_value = 0.0
        for (_, _, coef), c in zip(SEMI_BRUKNER_TERMS, counts):
            b_value += coef * correlator_from_counts(c)
        cb = 2.0
        qb = 2.0 * math.sqrt(2.0)
        achievement_ratio = (b_value - cb) / (qb - cb) if qb > cb else 0.0
        return SemiBruknerResult(
            expectation_value=b_value,
            achievement_ratio=achievement_ratio,
            lf_certified=b_value > cb,
        )

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list[CircuitBatch]:
        qubits = self._qubits()
        circuits = build_semi_brukner_circuits(qubits, device.num_qubits)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
