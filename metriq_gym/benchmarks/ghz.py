"""GHZ state preparation and verification benchmark.

Summary:
    Constructs a GHZ state on the device using a BFS spanning tree of the
    device's connectivity graph, then estimates fidelity via either direct
    fidelity estimation (DFE) or parity oscillation measurements.

Result interpretation:
    - population: probability of measuring all-zero or all-one bitstrings (Z basis).
    - coherence: off-diagonal element magnitude, measured via X-basis parity (DFE)
      or fitted amplitude of parity oscillation curve.
    - fidelity: (population + coherence) / 2, the GHZ state fidelity lower bound.

References:
    - Moses et al., "A Race-Track Trapped-Ion Quantum Processor",
      Phys. Rev. X 13, 041052 (2023). [arXiv:2305.03828]
    - Russo et al., "Compressed Sensing Verification of Large Entangled States",
      arXiv:2409.15302 (2024).
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit
from typing import TYPE_CHECKING

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


# ---------------------------------------------------------------------------
# Data / Result classes
# ---------------------------------------------------------------------------


@dataclass
class GHZData(BenchmarkData):
    num_qubits: int = 0
    method: str = "dfe"
    phases: list[float] = field(default_factory=list)
    num_flag_qubits: int = 0


class GHZResult(BenchmarkResult):
    population: BenchmarkScore = Field(description="Z-basis population of |0...0> + |1...1>")
    coherence: BenchmarkScore = Field(description="Off-diagonal coherence (X-basis parity or oscillation amplitude)")
    fidelity: BenchmarkScore = Field(description="GHZ state fidelity lower bound: (population + coherence) / 2")

    def compute_score(self) -> BenchmarkScore:
        return self.fidelity


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------


def _bfs_edges(graph, root: int, num_qubits: int) -> list[tuple[int, int]]:
    """BFS spanning tree edges from root, limited to num_qubits nodes."""
    visited = {root}
    queue = deque([root])
    edges: list[tuple[int, int]] = []

    while queue and len(visited) < num_qubits:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor not in visited and len(visited) < num_qubits:
                visited.add(neighbor)
                queue.append(neighbor)
                edges.append((node, neighbor))

    return edges


def _select_flag_qubits(
    graph, data_qubits: set[int], num_flags: int
) -> list[int]:
    """Select flag qubits: nodes adjacent to data qubits but not in the data set."""
    candidates = set()
    for dq in data_qubits:
        for neighbor in graph.neighbors(dq):
            if neighbor not in data_qubits:
                candidates.add(neighbor)
    return sorted(candidates)[:num_flags]


def build_ghz_circuits(
    graph,
    num_qubits: int,
    method: str = "dfe",
    phases: list[float] | None = None,
    num_flag_qubits: int = 0,
) -> tuple[list[QuantumCircuit], list[int], list[int]]:
    """Build GHZ verification circuits using BFS on the device connectivity graph.

    Returns:
        (circuits, data_qubits, flag_qubits) where circuits is the list of
        verification circuits and data/flag_qubits are the physical qubit indices used.
    """
    total_device_qubits = graph.num_nodes()

    # BFS from node 0 to select data qubits
    root = 0
    bfs = _bfs_edges(graph, root, num_qubits)
    data_qubits_set = {root}
    for ctrl, targ in bfs:
        data_qubits_set.add(ctrl)
        data_qubits_set.add(targ)
    data_qubits = sorted(data_qubits_set)[:num_qubits]

    # Select flag qubits from neighbors not in data set
    flag_qubits = _select_flag_qubits(graph, set(data_qubits), num_flag_qubits)

    n_total = total_device_qubits
    n_clbits = len(data_qubits) + len(flag_qubits)

    def _make_ghz_circuit() -> QuantumCircuit:
        """Create base GHZ state preparation circuit."""
        qc = QuantumCircuit(n_total, n_clbits)
        # Hadamard on root
        qc.h(data_qubits[0])
        # CNOT chain from BFS edges
        for ctrl, targ in bfs:
            qc.cx(ctrl, targ)
        qc.barrier()
        # ZZ flag operations
        for flag in flag_qubits:
            for neighbor in graph.neighbors(flag):
                if neighbor in set(data_qubits):
                    qc.cx(neighbor, flag)
        return qc

    def _add_measurements(qc: QuantumCircuit) -> None:
        """Add measurement gates for data and flag qubits."""
        qc.measure(data_qubits, list(range(len(data_qubits))))
        if flag_qubits:
            qc.measure(flag_qubits, list(range(len(data_qubits), n_clbits)))

    if method == "dfe":
        # Z-basis circuit
        z_qc = _make_ghz_circuit()
        _add_measurements(z_qc)

        # X-basis circuit
        x_qc = _make_ghz_circuit()
        for dq in data_qubits:
            x_qc.h(dq)
        _add_measurements(x_qc)

        return [z_qc, x_qc], data_qubits, flag_qubits

    elif method == "parity_oscillation":
        if phases is None:
            raise ValueError("phases required for parity_oscillation method")

        # Z-basis circuit
        z_qc = _make_ghz_circuit()
        _add_measurements(z_qc)

        # Oscillation circuits for each phase
        osc_circuits = []
        for phi in phases:
            osc_qc = _make_ghz_circuit()
            for dq in data_qubits:
                osc_qc.rz(phi, dq)
                osc_qc.h(dq)
            _add_measurements(osc_qc)
            osc_circuits.append(osc_qc)

        return [z_qc] + osc_circuits, data_qubits, flag_qubits

    else:
        raise ValueError(f"Unknown verification method: {method}")


# ---------------------------------------------------------------------------
# Post-selection and fidelity estimation
# ---------------------------------------------------------------------------


def post_select_results(counts: dict[str, int], num_flag_qubits: int) -> dict[str, int]:
    """Filter measurement results, keeping only outcomes where all flag qubits measure 0."""
    if num_flag_qubits == 0:
        return counts

    post_selected: dict[str, int] = {}
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        flags = cleaned[:num_flag_qubits]
        data = cleaned[num_flag_qubits:]
        if flags == "0" * num_flag_qubits:
            post_selected[data] = post_selected.get(data, 0) + count
    return post_selected


def estimate_fidelity_dfe(
    z_counts: dict[str, int],
    x_counts: dict[str, int],
    n: int,
    num_flag_qubits: int,
) -> tuple[float, float, float, float]:
    """Estimate GHZ fidelity using direct fidelity estimation.

    Returns: (population, coherence, population_err, coherence_err)
    """
    z_ps = post_select_results(z_counts, num_flag_qubits)
    x_ps = post_select_results(x_counts, num_flag_qubits)

    total_z = sum(z_ps.values())
    total_x = sum(x_ps.values())

    if total_z == 0 or total_x == 0:
        return 0.0, 0.0, 0.0, 0.0

    population = (z_ps.get("0" * n, 0) + z_ps.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    even_x = sum(c for b, c in x_ps.items() if b.count("1") % 2 == 0)
    coherence = (2 * even_x - total_x) / total_x
    c_err = np.sqrt((1 - coherence**2) / total_x)

    return population, coherence, p_err, c_err


def estimate_fidelity_oscillation(
    z_counts: dict[str, int],
    osc_counts_list: list[dict[str, int]],
    phases: list[float],
    n: int,
    num_flag_qubits: int,
) -> tuple[float, float, float, float]:
    """Estimate GHZ fidelity using parity oscillation curve fitting.

    Returns: (population, coherence, population_err, coherence_err)
    """
    z_ps = post_select_results(z_counts, num_flag_qubits)
    total_z = sum(z_ps.values())

    if total_z == 0:
        return 0.0, 0.0, 0.0, 0.0

    population = (z_ps.get("0" * n, 0) + z_ps.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    # Compute parities for each phase
    parities = []
    for osc_counts in osc_counts_list:
        ps = post_select_results(osc_counts, num_flag_qubits)
        total = sum(ps.values())
        if total == 0:
            parities.append(0.0)
            continue
        even = sum(c for b, c in ps.items() if b.count("1") % 2 == 0)
        parities.append((2 * even - total) / total)

    # Fit oscillation: amplitude * cos(n * phi + offset)
    def osc_func(phi, amplitude, offset):
        return amplitude * np.cos(n * phi + offset)

    try:
        popt, pcov = curve_fit(osc_func, np.array(phases), np.array(parities), p0=[0.5, 0])
        coherence = abs(popt[0])
        c_err = np.sqrt(pcov[0, 0])
    except Exception:
        coherence = 0.0
        c_err = 0.0

    return population, coherence, p_err, c_err


# ---------------------------------------------------------------------------
# Benchmark handler
# ---------------------------------------------------------------------------


class GHZBenchmark(Benchmark):
    def _build_circuits(self, device: "QuantumDevice") -> tuple[list[QuantumCircuit], list[int], list[int]]:
        graph = connectivity_graph(device)
        num_qubits = self.params.num_qubits
        method = getattr(self.params, "method", "dfe")
        num_flag_qubits = getattr(self.params, "num_flag_qubits", 0)

        phases = None
        if method == "parity_oscillation":
            num_phases = getattr(self.params, "num_phases", 20)
            phases = np.linspace(0, 2 * np.pi, num_phases, endpoint=False).tolist()

        return build_ghz_circuits(
            graph=graph,
            num_qubits=num_qubits,
            method=method,
            phases=phases,
            num_flag_qubits=num_flag_qubits,
        )

    def dispatch_handler(self, device: "QuantumDevice") -> GHZData:
        circuits, _data_qubits, _flag_qubits = self._build_circuits(device)
        method = getattr(self.params, "method", "dfe")
        num_flag_qubits = getattr(self.params, "num_flag_qubits", 0)

        phases: list[float] = []
        if method == "parity_oscillation":
            num_phases = getattr(self.params, "num_phases", 20)
            phases = np.linspace(0, 2 * np.pi, num_phases, endpoint=False).tolist()

        quantum_job = device.run(circuits, shots=self.params.shots)

        return GHZData.from_quantum_job(
            quantum_job,
            num_qubits=self.params.num_qubits,
            method=method,
            phases=phases,
            num_flag_qubits=num_flag_qubits,
        )

    def poll_handler(
        self,
        job_data: GHZData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> GHZResult:
        all_counts = flatten_counts(result_data)
        n = job_data.num_qubits
        method = job_data.method
        num_flags = job_data.num_flag_qubits

        if method == "dfe":
            z_counts, x_counts = all_counts[0], all_counts[1]
            pop, coh, p_err, c_err = estimate_fidelity_dfe(z_counts, x_counts, n, num_flags)
        else:
            z_counts = all_counts[0]
            osc_counts = list(all_counts[1:])
            pop, coh, p_err, c_err = estimate_fidelity_oscillation(
                z_counts, osc_counts, job_data.phases, n, num_flags
            )

        fidelity = (pop + coh) / 2
        f_err = np.sqrt(p_err**2 + c_err**2) / 2

        return GHZResult(
            population=BenchmarkScore(value=pop, uncertainty=p_err),
            coherence=BenchmarkScore(value=coh, uncertainty=c_err),
            fidelity=BenchmarkScore(value=fidelity, uncertainty=f_err),
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        circuits, _, _ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
