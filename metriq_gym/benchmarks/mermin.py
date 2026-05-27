"""Mermin / Belinskii-Klyshko benchmark implementation.

Summary:
    Multipartite generalisation of BSEQ. For each N in [min_n, max_n], the
    benchmark prepares an N-qubit GHZ state on a connected N-vertex subgraph
    of the device, measures the Belinskii-Klyshko polynomial M_N, and reports
    the estimated <M_N>. The single-scalar device summary is the
    ``mermin_depth``: the largest N for which the device's measured score
    still exceeds the LHV bound 2. A stricter ``gme_depth`` reports the
    largest N exceeding the biseparable bound 2^(N/2), certifying genuine
    multipartite entanglement.

Connectivity graph:
    For each N, the benchmark finds a connected N-vertex path on the device's
    native coupling graph via greedy BFS. Complete-graph devices (trapped-ion,
    simulators) trivially accommodate any N up to ``device.num_qubits``;
    sparse-topology devices stop at the longest connected path they can host.

Result interpretation:
    Polling returns MerminResult with:
        - score_per_n: dict of N -> measured <M_N>.
        - achievement_ratio_per_n: dict of N -> (score - 2) / (2^((N+1)/2) - 2).
          Positive values indicate a Mermin-Klyshko violation.
        - gme_certified_per_n: dict of N -> bool, true iff |score| > 2^(N/2).
        - mermin_depth: largest N for which achievement_ratio_per_n[N] > 0.
        - gme_depth: largest N for which gme_certified_per_n[N] is true.

References:
    - Belinskii & Klyshko, *Phys. Usp.* 36 (1993).
    - Mermin, *Phys. Rev. Lett.* 65, 1838 (1990).
    - Gisin & Bechmann-Pasquinucci, *Phys. Lett. A* 246 (1998) [biseparable bound].
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import rustworkx as rx
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.qplatform.device import connectivity_graph
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


def mermin_polynomial(n: int) -> dict[tuple[int, ...], float]:
    """Build the Belinskii-Klyshko polynomial M_N as a coefficient dict.

    Returns a mapping ``{(s_1, ..., s_n): c}`` where each ``s_j`` is ``0``
    (party j's unprimed observable A_j) or ``1`` (the primed A'_j), and ``c``
    is the polynomial coefficient. Zero-coefficient terms are pruned.

    For ``n == 2`` returns the CHSH coefficients ``{(0,0): 1, (0,1): 1,
    (1,0): 1, (1,1): -1}``.
    """
    if n < 2:
        raise ValueError(f"Mermin-Klyshko requires n >= 2, got {n}")
    poly: dict[tuple[int, ...], float] = {
        (0, 0): 1.0,
        (0, 1): 1.0,
        (1, 0): 1.0,
        (1, 1): -1.0,
    }
    for _ in range(3, n + 1):
        prev = poly
        prev_swapped = {tuple(1 - b for b in s): c for s, c in prev.items()}
        new: dict[tuple[int, ...], float] = {}
        for s, c in prev.items():
            new[s + (0,)] = new.get(s + (0,), 0.0) + 0.5 * c
            new[s + (1,)] = new.get(s + (1,), 0.0) + 0.5 * c
        for s, c in prev_swapped.items():
            new[s + (0,)] = new.get(s + (0,), 0.0) + 0.5 * c
            new[s + (1,)] = new.get(s + (1,), 0.0) - 0.5 * c
        poly = {s: c for s, c in new.items() if abs(c) > 1e-12}
    return poly


def optimal_angles(n: int) -> tuple[list[float], list[float]]:
    """Per-party angles saturating Tsirelson bound 2^((N+1)/2) on |GHZ_N>.

    Returns ``(theta, theta_prime)`` lists of length N. Party j uses the
    Hermitian operator with eigenvalues +/- 1 given by
    ``A(angle) = cos(angle) X + sin(angle) Y``.
    """
    if n < 2:
        raise ValueError(f"Mermin-Klyshko requires n >= 2, got {n}")
    theta = [-j * math.pi / (2 * n) for j in range(n)]
    theta_prime = [t + math.pi / 2 for t in theta]
    return theta, theta_prime


def find_connected_path(graph: rx.PyGraph, length: int) -> list[int] | None:
    """Find a connected path of ``length`` distinct vertices in ``graph``.

    Greedy BFS from each starting vertex: at each step, extend by any
    neighbour not already in the path. Returns the first such path found, or
    None if no path of the required length exists.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    num_nodes = graph.num_nodes()
    if length > num_nodes:
        return None
    for start in range(num_nodes):
        stack: deque[list[int]] = deque([[start]])
        while stack:
            path = stack.popleft()
            if len(path) == length:
                return path
            for neighbour in graph.neighbors(path[-1]):
                if neighbour not in path:
                    stack.append(path + [neighbour])
    return None


def prepare_ghz_path(num_qubits: int, qubits: list[int]) -> QuantumCircuit:
    """GHZ_N preparation along a path: Hadamard on the first qubit, CNOT ladder.

    Assumes ``qubits`` forms a connected path in the device coupling graph.
    """
    if len(qubits) < 2:
        raise ValueError(f"GHZ requires >= 2 qubits, got {len(qubits)}")
    qc = QuantumCircuit(num_qubits, name=f"GHZ{len(qubits)}")
    qc.h(qubits[0])
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
    return qc


def build_mermin_circuits(
    n: int,
    qubits: list[int],
    num_qubits: int,
) -> tuple[list[QuantumCircuit], list[tuple[int, ...]]]:
    """Build one measurement circuit per nonzero setting in ``mermin_polynomial(n)``.

    Each circuit prepares GHZ_N on ``qubits``, applies the per-qubit basis
    change ``RZ(angle); H`` so the subsequent Z-basis measurement implements
    ``A(angle) = cos(angle) X + sin(angle) Y``, and measures all N qubits
    into classical bits ``0..N-1``.

    Returns ``(circuits, settings)`` where ``settings[i]`` is the
    ``(s_1, ..., s_N)`` tuple corresponding to ``circuits[i]``, in canonical
    (lex-sorted) order.
    """
    if len(qubits) != n:
        raise ValueError(f"qubits length {len(qubits)} must equal n={n}")
    theta, theta_prime = optimal_angles(n)
    poly = mermin_polynomial(n)
    settings = sorted(poly.keys())
    circuits: list[QuantumCircuit] = []
    for setting in settings:
        qc = QuantumCircuit(num_qubits, n)
        qc.compose(prepare_ghz_path(num_qubits, qubits), inplace=True)
        for j, (s, q) in enumerate(zip(setting, qubits)):
            angle = theta[j] if s == 0 else theta_prime[j]
            qc.rz(angle, q)
            qc.h(q)
        for j, q in enumerate(qubits):
            qc.measure(q, j)
        circuits.append(qc)
    return circuits, settings


def correlator_from_counts(counts: dict[str, int], n: int) -> float:
    """Estimate <A_1 ... A_N> from a Z-basis count dict on N bits.

    The measured correlator is the average of (-1)^(b_1 + ... + b_N) over
    shots. Bitstrings outside the leading N bits are ignored.
    """
    total = 0
    weighted = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[-n:]
        parity = sum(int(b) for b in bits) % 2
        weighted += count * (1 if parity == 0 else -1)
        total += count
    if total == 0:
        return 0.0
    return weighted / total


class MerminResult(BenchmarkResult):
    score_per_n: dict[int, float]
    achievement_ratio_per_n: dict[int, float]
    gme_certified_per_n: dict[int, bool]
    mermin_depth: int
    gme_depth: int

    def _iter_metric_items(self):
        # Override to flatten the per-N dicts into scalar metrics for the
        # results.values / results.uncertainties accessors used by the
        # reporter, while keeping the dicts available on the model itself.
        for n, score in self.score_per_n.items():
            yield f"score_n{n}", float(score), None
        for n, ratio in self.achievement_ratio_per_n.items():
            yield f"achievement_ratio_n{n}", float(ratio), None
        yield "mermin_depth", float(self.mermin_depth), None
        yield "gme_depth", float(self.gme_depth), None

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=float(self.mermin_depth))


@dataclass
class MerminData(BenchmarkData):
    shots: int = 0
    min_n: int = 2
    max_n: int = 5
    num_qubits: int = 0
    paths_per_n: dict[int, list[int]] = field(default_factory=dict)
    settings_per_n: dict[int, list[list[int]]] = field(default_factory=dict)


def _settings_to_serialisable(settings: list[tuple[int, ...]]) -> list[list[int]]:
    return [list(s) for s in settings]


def _settings_from_serialisable(settings: list[list[int]]) -> list[tuple[int, ...]]:
    return [tuple(s) for s in settings]


class Mermin(Benchmark):
    """Benchmark class for the Mermin / Belinskii-Klyshko M_N test."""

    def _eligible_ns(self, topology_graph: rx.PyGraph) -> dict[int, list[int]]:
        min_n = self.params.min_n
        max_n = self.params.max_n
        if min_n < 2:
            raise ValueError(f"min_n must be >= 2, got {min_n}")
        if max_n < min_n:
            raise ValueError(f"max_n ({max_n}) must be >= min_n ({min_n})")
        paths: dict[int, list[int]] = {}
        for n in range(min_n, max_n + 1):
            path = find_connected_path(topology_graph, n)
            if path is None:
                break
            paths[n] = path
        if not paths:
            raise ValueError(
                f"No connected path of length {min_n} found in device coupling graph"
            )
        return paths

    def dispatch_handler(self, device: "QuantumDevice") -> MerminData:
        shots = self.params.shots
        topology_graph = connectivity_graph(device)
        paths = self._eligible_ns(topology_graph)
        num_qubits = device.num_qubits

        settings_per_n: dict[int, list[list[int]]] = {}
        all_circuit_groups: list[list[QuantumCircuit]] = []
        for n in sorted(paths.keys()):
            circuits, settings = build_mermin_circuits(n, paths[n], num_qubits)
            settings_per_n[n] = _settings_to_serialisable(settings)
            all_circuit_groups.append(circuits)

        quantum_jobs = [device.run(group, shots=shots) for group in all_circuit_groups]
        provider_job_ids = [
            job.id
            for group in quantum_jobs
            for job in (group if isinstance(group, list) else [group])
        ]

        return MerminData(
            provider_job_ids=provider_job_ids,
            shots=shots,
            min_n=self.params.min_n,
            max_n=self.params.max_n,
            num_qubits=num_qubits,
            paths_per_n=paths,
            settings_per_n=settings_per_n,
        )

    def poll_handler(
        self,
        job_data: MerminData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> MerminResult:
        counts_flat = flatten_counts(result_data)
        idx = 0
        score_per_n: dict[int, float] = {}
        achievement_ratio_per_n: dict[int, float] = {}
        gme_certified_per_n: dict[int, bool] = {}
        for n in sorted(job_data.settings_per_n.keys()):
            settings = _settings_from_serialisable(job_data.settings_per_n[n])
            poly = mermin_polynomial(n)
            score = 0.0
            for setting in settings:
                correlator = correlator_from_counts(counts_flat[idx], n)
                score += poly[setting] * correlator
                idx += 1
            score_per_n[n] = score
            cb = 2.0
            qb = 2.0 ** ((n + 1) / 2)
            achievement_ratio_per_n[n] = (score - cb) / (qb - cb) if qb > cb else 0.0
            gme_certified_per_n[n] = abs(score) > 2.0 ** (n / 2)

        violating = [n for n, r in achievement_ratio_per_n.items() if r > 0]
        certifying = [n for n, c in gme_certified_per_n.items() if c]
        mermin_depth = max(violating) if violating else 1
        gme_depth = max(certifying) if certifying else 1

        return MerminResult(
            score_per_n=score_per_n,
            achievement_ratio_per_n=achievement_ratio_per_n,
            gme_certified_per_n=gme_certified_per_n,
            mermin_depth=mermin_depth,
            gme_depth=gme_depth,
        )

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list[CircuitBatch]:
        topology_graph = connectivity_graph(device)
        paths = self._eligible_ns(topology_graph)
        num_qubits = device.num_qubits
        batches: list[CircuitBatch] = []
        for n in sorted(paths.keys()):
            circuits, _ = build_mermin_circuits(n, paths[n], num_qubits)
            batches.append(CircuitBatch(circuits=circuits, shots=self.params.shots))
        return batches
