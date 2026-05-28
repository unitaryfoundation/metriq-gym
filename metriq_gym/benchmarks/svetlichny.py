"""Svetlichny benchmark implementation.

Summary:
    Three-party benchmark evaluating the standard Svetlichny inequality on
    a GHZ_3 state shared across a chosen triple of qubits. The benchmark
    measures the Hermitian operator

        S_3 = A_1 B_1 C_1 + A_1 B_1 C_2 + A_1 B_2 C_1 - A_1 B_2 C_2
            + A_2 B_1 C_1 - A_2 B_1 C_2 - A_2 B_2 C_1 - A_2 B_2 C_2

    where each party j chooses between two equatorial-Pauli observables
    A_j(alpha_j) = cos(alpha_j) X + sin(alpha_j) Y with primed = unprimed
    + pi/2. The defining feature of Svetlichny is that the maximum of
    <S_3> over fully separable states coincides with the maximum over
    biseparable states, so violation certifies genuine tripartite
    entanglement rather than mere entanglement across some bipartition.

Bounds (on GHZ_3 with the canonical equatorial-Pauli parameterization):
    - Separable (and biseparable) bound: |S_3| <= 4.
    - Quantum maximum: 4*sqrt(2), achieved by GHZ_3 with the angles
      returned by ``optimal_angles``.

Result interpretation:
    Polling returns SvetlichnyResult with:
        - expectation_value: estimated <S_3>, in [-4*sqrt(2), 4*sqrt(2)]
          under ideal execution.
        - achievement_ratio: (|expectation_value| - 4) / (4*sqrt(2) - 4).
          Positive values indicate a Svetlichny violation.
        - gme_certified: bool, true iff |expectation_value| > 4. Genuine
          tripartite entanglement under the Svetlichny argument.

References:
    - Svetlichny, "Distinguishing three-body from two-body nonseparability
      by a Bell-type inequality", *Phys. Rev. D* 35, 3066 (1987).
    - Collins, Gisin, Popescu, Roberts, Scarani, "Bell-type inequalities
      to detect true n-body nonseparability", *Phys. Rev. Lett.* 88,
      170405 (2002).
"""

import math
from dataclasses import dataclass, field

from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.resource_estimation import CircuitBatch


# Coefficients of the canonical 3-party Svetlichny polynomial S_3 in the
# (s_1, s_2, s_3) basis with s_j in {0 (unprimed), 1 (primed)}.
SVETLICHNY_COEFFICIENTS: dict[tuple[int, int, int], float] = {
    (0, 0, 0): +1.0,
    (0, 0, 1): +1.0,
    (0, 1, 0): +1.0,
    (0, 1, 1): -1.0,
    (1, 0, 0): +1.0,
    (1, 0, 1): -1.0,
    (1, 1, 0): -1.0,
    (1, 1, 1): -1.0,
}

CLASSICAL_BOUND = 4.0
QUANTUM_BOUND = 4.0 * math.sqrt(2.0)


def optimal_angles() -> tuple[list[float], list[float]]:
    """Angles saturating |<S_3>| = 4*sqrt(2) on GHZ_3.

    Each party's unprimed angle is set to -pi/12; the primed angle is
    pi/2 above it. On GHZ_3, the correlator <A_1(s_1) A_2(s_2) A_3(s_3)>
    equals cos(alpha_1 + alpha_2 + alpha_3 + |s|*pi/2), and the chosen
    angles give sum alpha_j = -pi/4. Direct expansion then yields
    <S_3> = 4 cos(phi) - 4 sin(phi) = 4*sqrt(2) cos(phi + pi/4) with
    phi = -pi/4, which equals 4*sqrt(2).
    """
    theta = [-math.pi / 12] * 3
    theta_prime = [t + math.pi / 2 for t in theta]
    return theta, theta_prime


def prepare_ghz3(num_qubits: int, qubits: tuple[int, int, int]) -> QuantumCircuit:
    """GHZ_3 preparation on the chosen triple: H on the first, CNOT ladder."""
    qa, qb, qc = qubits
    qc_circ = QuantumCircuit(num_qubits, name="GHZ3")
    qc_circ.h(qa)
    qc_circ.cx(qa, qb)
    qc_circ.cx(qb, qc)
    return qc_circ


def build_svetlichny_circuits(
    qubits: tuple[int, int, int], num_qubits: int
) -> tuple[list[QuantumCircuit], list[tuple[int, int, int]]]:
    """Build the eight Svetlichny measurement circuits and the matching settings.

    Each circuit prepares GHZ_3 on ``qubits``, applies the per-qubit basis
    change ``RZ(angle); H`` so the subsequent Z-basis measurement
    implements the corresponding equatorial-Pauli observable, then
    measures all three qubits into classical bits ``0..2``. Returned in
    canonical (lex-sorted) order over ``SVETLICHNY_COEFFICIENTS`` keys.
    """
    theta, theta_prime = optimal_angles()
    settings = sorted(SVETLICHNY_COEFFICIENTS.keys())
    circuits: list[QuantumCircuit] = []
    for setting in settings:
        qc = QuantumCircuit(num_qubits, 3)
        qc.compose(prepare_ghz3(num_qubits, qubits), inplace=True)
        for j, (s, q) in enumerate(zip(setting, qubits)):
            angle = theta[j] if s == 0 else theta_prime[j]
            qc.rz(angle, q)
            qc.h(q)
        for j, q in enumerate(qubits):
            qc.measure(q, j)
        circuits.append(qc)
    return circuits, settings


def correlator_from_counts(counts: dict[str, int]) -> float:
    """Estimate <A_1 A_2 A_3> = <(-1)^(b_0 + b_1 + b_2)> from a 3-bit count dict."""
    total = 0
    weighted = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[-3:]
        parity = sum(int(b) for b in bits) % 2
        weighted += count * (1 if parity == 0 else -1)
        total += count
    if total == 0:
        return 0.0
    return weighted / total


class SvetlichnyResult(BenchmarkResult):
    expectation_value: float
    achievement_ratio: float
    gme_certified: bool

    def _iter_metric_items(self):
        yield "expectation_value", float(self.expectation_value), None
        yield "achievement_ratio", float(self.achievement_ratio), None
        yield "gme_certified", float(self.gme_certified), None

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=float(self.expectation_value))


@dataclass
class SvetlichnyData(BenchmarkData):
    shots: int = 0
    qubits: list[int] = field(default_factory=lambda: [0, 1, 2])
    num_qubits: int = 0


class Svetlichny(Benchmark):
    """Benchmark class for the 3-party Svetlichny inequality."""

    def _qubits(self) -> tuple[int, int, int]:
        qs = list(self.params.qubits)
        if len(qs) != 3:
            raise ValueError(f"Svetlichny requires exactly 3 qubits, got {qs}")
        if len(set(qs)) != 3:
            raise ValueError(f"Svetlichny requires three distinct qubits, got {qs}")
        return (qs[0], qs[1], qs[2])

    def dispatch_handler(self, device: QuantumDevice) -> SvetlichnyData:
        shots = self.params.shots
        qubits = self._qubits()
        num_qubits = device.num_qubits
        circuits, _ = build_svetlichny_circuits(qubits, num_qubits)
        return SvetlichnyData.from_quantum_job(
            device.run(circuits, shots=shots),
            shots=shots,
            qubits=list(qubits),
            num_qubits=num_qubits,
        )

    def poll_handler(
        self,
        job_data: SvetlichnyData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> SvetlichnyResult:
        counts = flatten_counts(result_data)
        settings = sorted(SVETLICHNY_COEFFICIENTS.keys())
        if len(counts) != len(settings):
            raise ValueError(f"expected {len(settings)} count dicts, got {len(counts)}")
        score = 0.0
        for setting, c in zip(settings, counts):
            score += SVETLICHNY_COEFFICIENTS[setting] * correlator_from_counts(c)
        gap = QUANTUM_BOUND - CLASSICAL_BOUND
        achievement_ratio = (abs(score) - CLASSICAL_BOUND) / gap if gap > 0 else 0.0
        return SvetlichnyResult(
            expectation_value=score,
            achievement_ratio=achievement_ratio,
            gme_certified=abs(score) > CLASSICAL_BOUND,
        )

    def estimate_resources_handler(
        self,
        device: QuantumDevice,
    ) -> list[CircuitBatch]:
        qubits = self._qubits()
        circuits, _ = build_svetlichny_circuits(qubits, device.num_qubits)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
