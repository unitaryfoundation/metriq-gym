"""EPLG (Error Per Layered Gate) benchmark implementation.

Summary:
    Measures layer fidelity across qubit chains using randomized benchmarking
    techniques. Computes EPLG scores at various chain lengths to characterize
    two-qubit gate performance across the device.

Result interpretation:
    Polling returns EPLGResult with:
        - chain_lengths: list of qubit chain lengths tested
        - chain_eplgs: EPLG values at each chain length
        - eplg_10/20/50/100: EPLG at standard reference points
        - score: average EPLG across reference points (lower is better)

References:
    - McKay et al., "Benchmarking quantum processor performance at scale"
      https://arxiv.org/abs/2311.05933
    - Based on qiskit-device-benchmarking layer fidelity notebook:
      https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/layer_fidelity.ipynb
"""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rustworkx as rx
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.qplatform.device import connectivity_graph
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob

@dataclass
class EPLGData(BenchmarkData):
    """Stores intermediate data between dispatch and poll."""

    num_qubits_in_chain: int
    qubit_chain: list[int]
    two_disjoint_layers: list[list[list[int]]]  # Serialized as nested lists
    lengths: list[int]
    num_samples: int
    shots: int
    seed: int
    two_qubit_gate: str
    one_qubit_basis_gates: list[str]


class EPLGResult(BenchmarkResult):
    """EPLG benchmark results."""

    chain_lengths: list[int]
    chain_eplgs: list[float]
    eplg_10: float | None = None
    eplg_20: float | None = None
    eplg_50: float | None = None
    eplg_100: float | None = None

    def compute_score(self) -> BenchmarkScore:
        """Compute average EPLG across reference points (10, 20, 50, 100 qubits)."""
        picked_vals = [
            v
            for v in [self.eplg_10, self.eplg_20, self.eplg_50, self.eplg_100]
            if v is not None
        ]
        if not picked_vals:
            return BenchmarkScore(value=0.0)
        # TODO: Propagate uncertainty from ProcessFidelity fits through EPLG calculation
        return BenchmarkScore(value=sum(picked_vals) / len(picked_vals))


# =============================================================================
# Utility Functions
# =============================================================================


def to_edges(path: list[int]) -> list[tuple[int, int]]:
    """Convert a path of nodes to a list of edges."""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def eplg_score_at_lengths(
    chain_lens: list[int],
    chain_eplgs: list[float],
    targets: list[int] | None = None,
) -> tuple[float, list[float], list[tuple[int, int]]]:
    """Compute EPLG score at specific qubit lengths.

    Args:
        chain_lens: List of chain lengths measured.
        chain_eplgs: EPLG values at each chain length.
        targets: Target qubit counts (default: [10, 20, 50, 100]).

    Returns:
        Tuple of (average_score, picked_values, picks) where picks shows
        which actual length was used for each target.
    """
    if targets is None:
        targets = [10, 20, 50, 100]

    idx = {length: i for i, length in enumerate(chain_lens)}
    picked_vals, picks = [], []

    for t in targets:
        if t in idx:
            picked_vals.append(chain_eplgs[idx[t]])
            picks.append((t, t))
        else:
            # Nearest-neighbor fallback when target length not available
            # TODO: Consider interpolation or curve fitting for more accurate estimates
            nearest = min(chain_lens, key=lambda x: (abs(x - t), x))
            picked_vals.append(chain_eplgs[idx[nearest]])
            picks.append((t, nearest))

    score = sum(picked_vals) / len(picked_vals) if picked_vals else 0.0
    return score, picked_vals, picks


def random_chain_complete_graph(
    num_qubits: int, length: int, seed: int | None = None
) -> list[int]:
    """Sample a random chain from a complete graph (all-to-all connectivity)."""
    rng = random.Random(seed)
    if length > num_qubits:
        raise ValueError(
            f"Chain length {length} cannot exceed number of qubits {num_qubits}"
        )
    return rng.sample(range(num_qubits), length)


def _allowed_edges(
    graph: rx.PyGraph,
    backend=None,
    twoq_gate: str | None = None,
    require_gate: bool = True,
) -> set[tuple[int, int]]:
    """Return set of undirected edges allowed for the chain.

    For IBM backends with require_gate=True, filters edges to only those
    supporting the specified two-qubit gate.
    """
    edges = [tuple(e) for e in graph.edge_list()]

    if not require_gate or backend is None or twoq_gate is None:
        return {tuple(sorted(e)) for e in edges}

    # IBM-specific: filter by gate availability
    try:
        gmap = backend.target[twoq_gate]
        allowed = set()
        for u, v in edges:
            if (u, v) in gmap or (v, u) in gmap:
                allowed.add(tuple(sorted((u, v))))
        return allowed
    except (AttributeError, KeyError):
        return {tuple(sorted(e)) for e in edges}


def random_chain_from_graph(
    graph: rx.PyGraph | rx.PyDiGraph,
    length: int,
    seed: int | None = None,
    backend=None,
    twoq_gate: str | None = None,
    require_gate: bool = True,
    restarts: int = 200,
) -> list[int]:
    """Sample a random simple path of given length from a graph.

    Args:
        graph: Connectivity graph (PyGraph or PyDiGraph).
        length: Desired chain length (number of nodes).
        seed: Random seed.
        backend: Optional Qiskit backend for gate filtering.
        twoq_gate: Two-qubit gate name for filtering.
        require_gate: If True and backend is provided, only use edges that
            support the specified two-qubit gate.
        restarts: Number of random restart attempts.

    Returns:
        List of qubit indices forming the chain.
    """
    rng = random.Random(seed)
    # Always work with undirected graph
    if hasattr(graph, "to_undirected"):
        graph_und = graph.to_undirected(multigraph=False)
    else:
        graph_und = graph

    allowed = _allowed_edges(graph_und, backend, twoq_gate, require_gate)
    if not allowed:
        raise RuntimeError("No allowed 2-qubit edges found to form a chain.")

    n_nodes = graph_und.num_nodes()
    adj = {i: [] for i in range(n_nodes)}
    for u, v in allowed:
        adj[u].append(v)
        adj[v].append(u)

    allowed_edges_list = list(allowed)

    for _ in range(restarts):
        u, v = rng.choice(allowed_edges_list)
        path = [u, v]
        used = {u, v}

        while len(path) < length:
            extended = False
            for side in rng.sample(["head", "tail"], 2):
                if side == "head":
                    cur = path[0]
                    cands = [w for w in adj[cur] if w not in used]
                    if cands:
                        w = rng.choice(cands)
                        path.insert(0, w)
                        used.add(w)
                        extended = True
                        break
                else:
                    cur = path[-1]
                    cands = [w for w in adj[cur] if w not in used]
                    if cands:
                        w = rng.choice(cands)
                        path.append(w)
                        used.add(w)
                        extended = True
                        break
            if not extended:
                break

        if len(path) == length:
            return path

    raise RuntimeError(
        f"Failed to sample chain of length={length} after {restarts} restarts."
    )


# =============================================================================
# IBM-specific Chain Selection
# =============================================================================


def _is_ibm_backend(device: "QuantumDevice") -> bool:
    """Check if device is an IBM/Qiskit backend."""
    try:
        from qbraid.runtime import QiskitBackend

        return isinstance(device, QiskitBackend)
    except ImportError:
        return False


def _get_ibm_backend(device: "QuantumDevice"):
    """Extract the underlying Qiskit backend from a QBraid device."""
    if _is_ibm_backend(device):
        return device._backend
    return None


def _pick_twoq_gate(backend, prefer: str | None = None) -> str | None:
    """Pick a two-qubit gate from an IBM backend."""
    if prefer and prefer in getattr(backend.target, "operation_names", []):
        return prefer
    for g in ("ecr", "cx", "cz", "iswap", "rxx", "xx_plus_yy", "xx", "ms"):
        if g in getattr(backend.target, "operation_names", []):
            return g
    return None


def _path_fidelity(
    path: list[int], backend, twoq_gate: str, correct_by_duration: bool = True
) -> float:
    """Compute estimated fidelity of 2-qubit gates on a path (IBM-specific)."""

    def to_edges_ibm(p):
        edges = []
        prev = None
        for node in p:
            if prev is not None:
                g = backend.target.build_coupling_map(twoq_gate).graph
                if g.has_edge(prev, node):
                    edges.append((prev, node))
                else:
                    edges.append((node, prev))
            prev = node
        return edges

    path_edges = to_edges_ibm(path)
    max_duration = max(backend.target[twoq_gate][qs].duration for qs in path_edges)

    def gate_fidelity(qpair):
        duration = backend.target[twoq_gate][qpair].duration
        scale = max_duration / duration if correct_by_duration else 1.0
        return max(0.25, 1 - (1.25 * backend.target[twoq_gate][qpair].error)) ** scale

    return float(np.prod([gate_fidelity(qs) for qs in path_edges]))


def _flatten_paths(paths, cutoff: int | None = None) -> list[list[int]]:
    """Flatten paths dictionary from rustworkx all_pairs_all_simple_paths."""
    return [
        path
        for s, s_paths in paths.items()
        for t, st_paths in s_paths.items()
        for path in st_paths[:cutoff]
        if s < t
    ]


def select_best_chain_ibm(
    backend, twoq_gate: str, num_qubits_in_chain: int
) -> list[int]:
    """Select the best qubit chain for an IBM backend based on calibration data.

    Searches all simple paths of the required length and returns the one with
    highest predicted layer fidelity.
    """
    coupling_map = backend.target.build_coupling_map(twoq_gate)
    graph = coupling_map.graph

    # Always work with undirected graph
    if hasattr(graph, "to_undirected"):
        graph_und = graph.to_undirected(multigraph=False)
    else:
        graph_und = graph

    paths = rx.all_pairs_all_simple_paths(
        graph_und,
        min_depth=num_qubits_in_chain,
        cutoff=num_qubits_in_chain,
    )
    paths = _flatten_paths(paths, cutoff=400)

    if not paths:
        raise ValueError(
            f"No qubit chain with length={num_qubits_in_chain} exists. "
            "Try smaller num_qubits_in_chain."
        )

    return max(paths, key=lambda p: _path_fidelity(p, backend, twoq_gate))


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_eplg_results(
    exp_data,
    two_disjoint_layers: list[list[tuple[int, int]]],
    qubit_chain: list[int],
) -> tuple[list[int], list[float]] | tuple[None, None]:
    """Run EPLG analysis on experiment data.

    Args:
        exp_data: qiskit_experiments ExperimentData object.
        two_disjoint_layers: The two disjoint edge layers.
        qubit_chain: The qubit chain used.

    Returns:
        Tuple of (chain_lengths, chain_eplgs) or (None, None) if analysis fails.
    """
    import time

    exp_data.block_for_results()
    exp_data.experiment.analysis.run(exp_data)

    # Wait for analysis
    max_wait = 60
    waited = 0
    status = str(exp_data.analysis_status())
    while "RUNNING" in status and waited < max_wait:
        time.sleep(0.5)
        waited += 0.5
        status = str(exp_data.analysis_status())

    df = exp_data.analysis_results(dataframe=True)
    pfdf = df[df.name == "ProcessFidelity"]
    pfdf = pfdf.fillna({"value": 0})

    # Compute LF by chain length
    lf_sets = two_disjoint_layers
    full_layer = [None] * (len(lf_sets[0]) + len(lf_sets[1]))
    full_layer[::2] = lf_sets[0]
    full_layer[1::2] = lf_sets[1]
    full_layer = [(qubit_chain[0],)] + full_layer + [(qubit_chain[-1],)]

    pfs = []
    for qubits in full_layer:
        matching = pfdf[pfdf.qubits == qubits]
        if matching.empty:
            pfs.append(0)
        else:
            pfs.append(matching.iloc[0]["value"])

    pfs = [x.n if hasattr(x, "n") and x != 0 else (x if x != 0 else 0) for x in pfs]
    if len(pfs) >= 2:
        pfs[0] = pfs[0] ** 2
        pfs[-1] = pfs[-1] ** 2

    chain_lens = list(range(4, len(pfs), 2))
    if not chain_lens:
        return None, None

    chain_fids = []
    for length in chain_lens:
        w = length + 1
        fid_w = max(
            np.sqrt(pfs[s]) * np.prod(pfs[s + 1 : s + w - 1]) * np.sqrt(pfs[s + w - 1])
            for s in range(len(pfs) - w + 1)
        )
        chain_fids.append(fid_w)

    num_2q_gates = [length - 1 for length in chain_lens]
    chain_eplgs = [
        1 - (fid ** (1 / num_2q)) if num_2q > 0 else 0.0
        for num_2q, fid in zip(num_2q_gates, chain_fids)
    ]

    return chain_lens, [float(x) for x in chain_eplgs]


# =============================================================================
# Benchmark Class
# =============================================================================


class EPLG(Benchmark[EPLGData, EPLGResult]):
    """EPLG (Error Per Layered Gate) benchmark."""

    def _select_qubit_chain(
        self, device: "QuantumDevice"
    ) -> tuple[list[int], str, list[str]]:
        """Select qubit chain based on device type and parameters.

        Returns:
            Tuple of (qubit_chain, two_qubit_gate, one_qubit_basis_gates).
        """
        num_qubits_in_chain = self.params.num_qubits_in_chain
        chain_type = self.params.chain_type
        two_qubit_gate = self.params.two_qubit_gate
        one_qubit_basis_gates = self.params.one_qubit_basis_gates
        seed = self.params.seed

        ibm_backend = _get_ibm_backend(device)

        if ibm_backend is not None:
            # IBM device: auto-detect gate if not specified
            detected_gate = _pick_twoq_gate(ibm_backend, two_qubit_gate)
            if detected_gate is not None:
                two_qubit_gate = detected_gate
            # two_qubit_gate now has either the detected gate or the schema default

            if chain_type == "best":
                qubit_chain = select_best_chain_ibm(
                    ibm_backend, two_qubit_gate, num_qubits_in_chain
                )
            else:
                coupling_map = ibm_backend.target.build_coupling_map(two_qubit_gate)
                qubit_chain = random_chain_from_graph(
                    coupling_map.graph,
                    num_qubits_in_chain,
                    seed=seed,
                    backend=ibm_backend,
                    twoq_gate=two_qubit_gate,
                    require_gate=False,
                )
        else:
            # Non-IBM device: use connectivity graph
            graph = connectivity_graph(device)
            num_qubits = graph.num_nodes()

            # Check if it's a complete graph (all-to-all)
            expected_edges = num_qubits * (num_qubits - 1) // 2
            is_complete = graph.num_edges() >= expected_edges

            if is_complete:
                # For fully connected devices, use contiguous qubit indices
                qubit_chain = list(range(num_qubits_in_chain))
            else:
                qubit_chain = random_chain_from_graph(
                    graph, num_qubits_in_chain, seed=seed
                )

        return qubit_chain, two_qubit_gate, list(one_qubit_basis_gates)

    def _build_circuits(
        self, device: "QuantumDevice"
    ) -> tuple[list, list[int], list[list[tuple[int, int]]], str, list[str]]:
        """Build LayerFidelity circuits.

        Returns:
            Tuple of (circuits, qubit_chain, two_disjoint_layers,
                      two_qubit_gate, one_qubit_basis_gates).
        """
        qubit_chain, two_qubit_gate, one_qubit_basis_gates = self._select_qubit_chain(
            device
        )

        lengths = self.params.lengths
        num_samples = self.params.num_samples
        seed = self.params.seed

        all_pairs = to_edges(qubit_chain)
        two_disjoint_layers = [all_pairs[0::2], all_pairs[1::2]]

        # Create LayerFidelity experiment with explicit gate specification
        # (gates were already determined/auto-detected in _select_qubit_chain)
        lfexp = LayerFidelity(
            physical_qubits=qubit_chain,
            two_qubit_layers=two_disjoint_layers,
            lengths=lengths,
            num_samples=num_samples,
            seed=seed,
            two_qubit_gate=two_qubit_gate,
            one_qubit_basis_gates=one_qubit_basis_gates,
        )

        lfexp.experiment_options.max_circuits = 2 * num_samples * len(lengths)
        circuits = lfexp.circuits()

        return circuits, qubit_chain, two_disjoint_layers, two_qubit_gate, one_qubit_basis_gates

    def dispatch_handler(self, device: "QuantumDevice") -> EPLGData:
        """Generate circuits, submit to device, return EPLGData."""
        circuits, qubit_chain, two_disjoint_layers, two_qubit_gate, one_qubit_basis_gates = (
            self._build_circuits(device)
        )

        shots = self.params.shots
        quantum_job = device.run(circuits, shots=shots)

        # Serialize two_disjoint_layers as nested lists
        serialized_layers = [
            [list(edge) for edge in layer] for layer in two_disjoint_layers
        ]

        return EPLGData.from_quantum_job(
            quantum_job,
            num_qubits_in_chain=self.params.num_qubits_in_chain,
            qubit_chain=qubit_chain,
            two_disjoint_layers=serialized_layers,
            lengths=self.params.lengths,
            num_samples=self.params.num_samples,
            shots=shots,
            seed=self.params.seed,
            two_qubit_gate=two_qubit_gate,
            one_qubit_basis_gates=one_qubit_basis_gates,
        )

    def poll_handler(
        self,
        job_data: EPLGData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> EPLGResult:
        """Process results and compute EPLG metrics."""
        from qiskit.result import Result as QiskitResult
        from qiskit.result.models import ExperimentResult, ExperimentResultData
        from qiskit_experiments.framework import ExperimentData

        from metriq_gym.helpers.task_helpers import flatten_counts

        # Deserialize two_disjoint_layers back to tuples
        two_disjoint_layers = [
            [tuple(edge) for edge in layer] for layer in job_data.two_disjoint_layers
        ]

        # Recreate the LayerFidelity experiment
        lfexp = LayerFidelity(
            physical_qubits=job_data.qubit_chain,
            two_qubit_layers=two_disjoint_layers,
            lengths=job_data.lengths,
            num_samples=job_data.num_samples,
            seed=job_data.seed,
            two_qubit_gate=job_data.two_qubit_gate,
            one_qubit_basis_gates=job_data.one_qubit_basis_gates,
        )
        lfexp.experiment_options.max_circuits = (
            2 * job_data.num_samples * len(job_data.lengths)
        )

        # Get original circuits to extract metadata
        original_circuits = lfexp.circuits()
        counts_list = flatten_counts(result_data)

        # Build Qiskit Result from counts
        experiment_results = []
        for i, counts in enumerate(counts_list):
            metadata = original_circuits[i].metadata if i < len(original_circuits) else {}
            exp_result = ExperimentResult(
                shots=job_data.shots,
                success=True,
                data=ExperimentResultData(counts=counts),
                header={"name": f"circuit_{i}", "metadata": metadata},
            )
            experiment_results.append(exp_result)

        # backend_name is arbitrary - only used for Result object construction,
        # not for analysis
        qiskit_result = QiskitResult(
            backend_name="qbraid_device",
            backend_version="1.0",
            job_id="eplg_job",
            success=True,
            results=experiment_results,
        )

        # Create ExperimentData and add results
        exp_data = ExperimentData(experiment=lfexp)
        exp_data.add_data(qiskit_result)
        exp_data.auto_save = False

        # Run analysis
        chain_lens, chain_eplgs = analyze_eplg_results(
            exp_data, two_disjoint_layers, job_data.qubit_chain
        )

        if chain_lens is None or chain_eplgs is None:
            return EPLGResult(
                chain_lengths=[],
                chain_eplgs=[],
            )

        # Compute EPLG at reference points
        _, picked_vals, picks = eplg_score_at_lengths(chain_lens, chain_eplgs)

        return EPLGResult(
            chain_lengths=chain_lens,
            chain_eplgs=chain_eplgs,
            eplg_10=picked_vals[0] if len(picked_vals) > 0 else None,
            eplg_20=picked_vals[1] if len(picked_vals) > 1 else None,
            eplg_50=picked_vals[2] if len(picked_vals) > 2 else None,
            eplg_100=picked_vals[3] if len(picked_vals) > 3 else None,
        )

    def estimate_resources_handler(
        self, device: "QuantumDevice"
    ) -> list[CircuitBatch]:
        """Return circuit batches for resource estimation."""
        circuits, _, _, _, _ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
