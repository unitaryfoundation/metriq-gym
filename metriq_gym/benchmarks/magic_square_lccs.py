"""Magic square LCCS benchmark.

Summary:
    Device-level analogue of BSEQ for the magic-square / LCS family.
    Enumerates every 4-cycle (2x2 plaquette) of the device coupling
    graph, runs the four-qubit Mermin--Peres magic-square benchmark on
    each, and aggregates passing 4-cycles to a largest-connected-
    component-size (LCCS) metric on the qubits they cover.

    Each 4-cycle (a1, a2, b1, b2) is a 4-vertex induced cycle in the
    coupling graph with all four magic-square edges natively present:
    Bell-pair edges a1-b1 and a2-b2, plus the row-3 / col-3
    basis-change edges a1-a2 and b1-b2. A 4-cycle is **passing** iff
    its empirical magic-square win probability strictly exceeds the
    classical bound beta = 8/9.

Topology dependence:
    Heavy-hex coupling graphs have girth 6 and very few 4-cycles, so
    the metric typically collapses to LCCS <= 4 (or 0) there. Square
    lattice topologies have one 4-cycle per 2x2 plaquette and produce
    a meaningful LCCS that scales with device size. All-to-all
    connectivity saturates the metric at the device qubit count up to
    noise.

Result interpretation:
    Polling returns MagicSquareLCCSResult with:
        - lccs: size of the largest connected component of qubits that
          belong to at least one passing 4-cycle (computed in the
          induced subgraph of the device coupling graph).
        - num_cycles_tested: number of 4-cycles enumerated.
        - num_passing: number of 4-cycles whose win probability
          exceeds the win_threshold.

References:
    - Cosentino et al. (Metriq) for the BSEQ aggregation pattern.
    - Mermin (PRL 65, 3373, 1990) and Peres (Phys. Lett. A 151, 107,
      1990) for the magic-square game itself.
"""

from dataclasses import dataclass, field

import rustworkx as rx
from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.benchmarks.magic_square import (
    CLASSICAL_BOUND,
    build_magic_square_circuits,
    win_probability_from_counts,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.qplatform.device import connectivity_graph
from metriq_gym.resource_estimation import CircuitBatch


def enumerate_4cycles(graph: rx.PyGraph) -> list[tuple[int, int, int, int]]:
    """Enumerate every 4-cycle in ``graph``, deduplicated by vertex set.

    Each returned tuple ``(a1, a2, b1, b2)`` is a labeling that satisfies
    the magic-square edge requirements: a1-b1, a2-b2 (Bell-pair edges),
    a1-a2 (row 3 basis change), b1-b2 (col 3 basis change). A given
    4-vertex induced cycle is returned once, with a canonical labeling
    chosen by lexicographic minimisation over the four rotational and
    Alice/Bob-swap variants.
    """
    seen: set[frozenset[int]] = set()
    cycles: list[tuple[int, int, int, int]] = []
    for u, v in graph.edge_list():
        u_nbrs = set(graph.neighbors(u)) - {v}
        v_nbrs = set(graph.neighbors(v)) - {u}
        for x in u_nbrs:
            for y in v_nbrs:
                if x == y or x == v or y == u:
                    continue
                if not graph.has_edge(x, y):
                    continue
                vset = frozenset({u, v, x, y})
                if vset in seen:
                    continue
                seen.add(vset)
                # Canonicalise: alice = {u, x}, bob = {v, y}, with the
                # smaller qubit going first in each pair, and Alice
                # being the pair with the smaller minimum.
                alice = tuple(sorted((u, x)))
                bob = tuple(sorted((v, y)))
                if min(alice) > min(bob):
                    alice, bob = bob, alice
                cycles.append((alice[0], alice[1], bob[0], bob[1]))
    return cycles


def compute_lccs(
    passing_cycles: list[tuple[int, int, int, int]],
    graph: rx.PyGraph,
) -> int:
    """Largest connected component (in qubits) of the union of passing 4-cycles.

    A qubit is *covered* if it appears in at least one passing 4-cycle.
    The metric is the size of the largest connected component, in the
    induced subgraph of ``graph`` on the covered qubits.
    """
    if not passing_cycles:
        return 0
    covered: set[int] = set()
    for cycle in passing_cycles:
        covered.update(cycle)
    sub = graph.subgraph(sorted(covered))
    components = rx.connected_components(sub)
    return max((len(comp) for comp in components), default=0)


def _graph_from_edge_list(num_qubits: int, edge_list: list[list[int]]) -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(list(range(num_qubits)))
    for u, v in edge_list:
        if not g.has_edge(u, v):
            g.add_edge(u, v, None)
    return g


class MagicSquareLCCSResult(BenchmarkResult):
    lccs: int
    num_cycles_tested: int
    num_passing: int
    win_threshold: float

    def _iter_metric_items(self):
        yield "lccs", float(self.lccs), None
        yield "num_cycles_tested", float(self.num_cycles_tested), None
        yield "num_passing", float(self.num_passing), None

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=float(self.lccs))


@dataclass
class MagicSquareLCCSData(BenchmarkData):
    shots: int = 0
    cycles: list[list[int]] = field(default_factory=list)
    edge_list: list[list[int]] = field(default_factory=list)
    win_threshold: float = CLASSICAL_BOUND
    num_qubits: int = 0


class MagicSquareLCCS(Benchmark):
    """Magic-square LCCS benchmark on the device's 4-cycle skeleton."""

    def _win_threshold(self) -> float:
        threshold = getattr(self.params, "win_threshold", None)
        return CLASSICAL_BOUND if threshold is None else float(threshold)

    def dispatch_handler(self, device: QuantumDevice) -> MagicSquareLCCSData:
        shots = self.params.shots
        graph = connectivity_graph(device)
        cycles = enumerate_4cycles(graph)
        if not cycles:
            raise ValueError(
                "device coupling graph contains no 4-cycles; magic-square LCCS "
                "is undefined for this topology."
            )
        num_qubits = device.num_qubits
        edge_list = [list(e) for e in graph.edge_list()]
        all_circuits: list[QuantumCircuit] = []
        for a1, a2, b1, b2 in cycles:
            all_circuits.extend(
                build_magic_square_circuits(
                    alice_qubits=(a1, a2),
                    bob_qubits=(b1, b2),
                    num_qubits=num_qubits,
                )
            )
        return MagicSquareLCCSData.from_quantum_job(
            device.run(all_circuits, shots=shots),
            shots=shots,
            cycles=[list(c) for c in cycles],
            edge_list=edge_list,
            win_threshold=self._win_threshold(),
            num_qubits=num_qubits,
        )

    def poll_handler(
        self,
        job_data: MagicSquareLCCSData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> MagicSquareLCCSResult:
        counts_list = flatten_counts(result_data)
        cycles = [tuple(c) for c in job_data.cycles]
        if len(counts_list) != 9 * len(cycles):
            raise ValueError(
                f"expected {9 * len(cycles)} count dicts, got {len(counts_list)}"
            )
        passing: list[tuple[int, int, int, int]] = []
        for i, cycle in enumerate(cycles):
            cycle_counts = counts_list[9 * i : 9 * (i + 1)]
            wins = []
            idx = 0
            for r in (1, 2, 3):
                for c in (1, 2, 3):
                    wins.append(win_probability_from_counts(cycle_counts[idx], r, c))
                    idx += 1
            avg = sum(wins) / 9.0
            if avg > job_data.win_threshold:
                passing.append(cycle)  # type: ignore[arg-type]
        graph = _graph_from_edge_list(job_data.num_qubits, job_data.edge_list)
        lccs = compute_lccs(passing, graph)
        return MagicSquareLCCSResult(
            lccs=lccs,
            num_cycles_tested=len(cycles),
            num_passing=len(passing),
            win_threshold=job_data.win_threshold,
        )

    def estimate_resources_handler(
        self,
        device: QuantumDevice,
    ) -> list[CircuitBatch]:
        graph = connectivity_graph(device)
        cycles = enumerate_4cycles(graph)
        num_qubits = device.num_qubits
        all_circuits: list[QuantumCircuit] = []
        for a1, a2, b1, b2 in cycles:
            all_circuits.extend(
                build_magic_square_circuits(
                    alice_qubits=(a1, a2),
                    bob_qubits=(b1, b2),
                    num_qubits=num_qubits,
                )
            )
        return [CircuitBatch(circuits=all_circuits, shots=self.params.shots)]
