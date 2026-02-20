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
    - [McKay et al., "Benchmarking quantum processor performance at scale",
      arXiv:2311.05933](https://arxiv.org/abs/2311.05933).
    - Based on [qiskit-device-benchmarking layer fidelity notebook](https://github.com/qiskit-community/qiskit-device-benchmarking/blob/main/notebooks/layer_fidelity.ipynb).
"""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rustworkx as rx
import time
import warnings
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qiskit.result import Result as QiskitResult
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit_experiments.framework import ExperimentData

from metriq_gym.helpers.task_helpers import flatten_counts

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
)
from metriq_gym.qplatform.device import connectivity_graph, connectivity_graph_for_gate
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


@dataclass
class EPLGData(BenchmarkData):
    """Stores intermediate data between dispatch and poll."""

    num_qubits_in_chain: int
    qubit_chain: list[int]  # Indices of qubits forming the chain
    two_disjoint_layers: list[list[list[int]]]  # List of disjoint edges along the chain
    lengths: list[int]  # Lengths of RB protocol to test
    num_samples: int  # Number of random benchmark samples per length
    shots: int  # Number of shots per circuit
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


# =============================================================================
# Utility Functions
# =============================================================================


def random_chain_from_graph(
    graph: rx.PyGraph,
    length: int,
    seed: int | None = None,
    restarts: int = 200,
) -> list[int]:
    """Sample a random simple path of given length from a graph.

    Args:
        graph: Connectivity graph (undirected).
        length: Desired chain length (number of nodes).
        seed: Random seed.
        restarts: Number of random restart attempts.

    Returns:
        List of qubit indices forming the chain.
    """
    rng = random.Random(seed)

    allowed = {tuple(sorted(e)) for e in graph.edge_list()}

    n_nodes = graph.num_nodes()
    adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
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

    raise RuntimeError(f"Failed to sample chain of length={length} after {restarts} restarts.")


# =============================================================================
# Analysis Functions
# =============================================================================

def tune_layer_fidelity_fits_soft(
    lfexp: LayerFidelity,
    b_half_width: float = 0.05,
) -> LayerFidelity:
    """Soft-constrain the RB offset parameter ``b`` inside LayerFidelity analysis.

    This is intended for slow-decay devices (e.g., trapped-ion) when the RB depth
    schedule is truncated and the tail/asymptote is poorly resolved. We stabilize
    the fit by bounding ``b`` near its expected baseline value ``1/2**n``, where
    ``n`` is the number of qubits in the RB survival probability outcome.

    Args:
        lfexp: A qiskit-experiments LayerFidelity instance.
        b_half_width: Half-width for the interval around ``1/2**n`` used to bound ``b``.

    Returns:
        The same LayerFidelity instance, with updated analysis options.
    """
    if b_half_width < 0:
        raise ValueError("b_half_width must be non-negative.")

    analysis = getattr(lfexp, "analysis", None)
    if analysis is None:
        return lfexp

    comp = getattr(analysis, "component_analysis", None)
    if comp is None:
        return lfexp

    # Only warn once per call to avoid spamming if multiple RBAnalysis objects fail.
    warned = False

    try:
        single_layer_analyses = comp()
    except Exception as exc:
        warnings.warn(
            f"Failed to traverse LayerFidelity analysis tree (soft b-constraint not applied): {exc}",
            category=RuntimeWarning,
        )
        return lfexp

    # LayerFidelityAnalysis -> SingleLayerFidelityAnalysis -> RBAnalysis
    for single_layer_analysis in single_layer_analyses:
        rb_comp = getattr(single_layer_analysis, "component_analysis", None)
        if rb_comp is None:
            continue

        for rb_analysis in rb_comp():
            opts = getattr(rb_analysis, "options", None)
            outcome = getattr(opts, "outcome", None)
            if not outcome:
                continue

            n = len(outcome)
            b0 = 1.0 / (2**n)
            lo = max(0.0, b0 - b_half_width)
            hi = min(1.0, b0 + b_half_width)

            bounds = dict(getattr(opts, "bounds", {}) or {})
            bounds["b"] = (lo, hi)

            p0 = dict(getattr(opts, "p0", {}) or {})
            p0.setdefault("b", b0)

            try:
                rb_analysis.set_options(bounds=bounds, p0=p0)
            except Exception as exc:
                if not warned:
                    warnings.warn(
                        f"Failed to apply soft b-constraint to an RBAnalysis: {exc}",
                        category=RuntimeWarning,
                    )
                    warned = True

    return lfexp

def analyze_eplg_results(
    exp_data: ExperimentData,
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

    exp_data.block_for_results()
    exp_data.experiment.analysis.run(exp_data)

    # Wait for analysis
    max_wait = 60
    waited: float = 0
    status = str(exp_data.analysis_status())
    while "RUNNING" in status and waited < max_wait:
        time.sleep(0.5)
        waited += 0.5
        status = str(exp_data.analysis_status())

    df = exp_data.analysis_results(dataframe=True)
    pfdf = df[df.name == "ProcessFidelity"]
    pfdf = pfdf.fillna({"value": 0})

    # Build full layer by interleaving the disjoint layers
    lf_sets, lf_qubits = two_disjoint_layers, qubit_chain
    full_layer: list[tuple[int, ...] | None] = [None] * (len(lf_sets[0]) + len(lf_sets[1]))
    full_layer[::2] = lf_sets[0]
    full_layer[1::2] = lf_sets[1]
    full_layer = [(lf_qubits[0],)] + full_layer + [(lf_qubits[-1],)]

    # Collect process fidelities for each segment in the full layer; any missing failed
    # to fit and will be assigned a value of 0
    pf_map = pfdf.set_index("qubits")["value"]
    # Reindex without filling to detect gaps and emit a warning
    pfs_series = pf_map.reindex(full_layer)
    missing_idx = pfs_series[pfs_series.isna()].index.tolist()
    if missing_idx:
        warnings.warn(
            f"Missing ProcessFidelity for {len(missing_idx)} segment(s); filling with 0. Qubits missing {missing_idx}",
            category=RuntimeWarning,
            stacklevel=2,
        )
    pfs = pfs_series.fillna(0).tolist()
    pfs = list(map(lambda x: x.n if x != 0 else 0, pfs))
    pfs[0] = pfs[0] ** 2
    pfs[-1] = pfs[-1] ** 2

    chain_lens = list(range(4, len(pfs), 2))

    if len(chain_lens) == 0:
        warnings.warn(
            f"Chain too short for EPLG analysis; has {len(pfs)} process fidelities; need at least 5.",
            category=RuntimeWarning,
            stacklevel=2,
        )
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
    chain_eplgs = [1 - (fid ** (1 / num_2q)) for num_2q, fid in zip(num_2q_gates, chain_fids)]

    return chain_lens, [float(x) for x in chain_eplgs]


# =============================================================================
# Benchmark Class
# =============================================================================


class EPLG(Benchmark[EPLGData, EPLGResult]):
    """EPLG (Error Per Layered Gate) benchmark."""

    def _build_circuits(
        self, device: "QuantumDevice"
    ) -> tuple[list, list[int], list[list[tuple[int, int]]], str, list[str]]:
        """Build LayerFidelity circuits.

        Returns:
            Tuple of (circuits, qubit_chain, two_disjoint_layers,
                      two_qubit_gate, one_qubit_basis_gates).
        """
        num_qubits_in_chain = self.params.num_qubits_in_chain
        two_qubit_gate = self.params.two_qubit_gate
        one_qubit_basis_gates = self.params.one_qubit_basis_gates
        lengths = self.params.lengths
        num_samples = self.params.num_samples
        seed = self.params.seed

        # If the device has restricted connectivity for the 2 gate, use
        # that restricted topology to create the chain
        graph = connectivity_graph_for_gate(device, two_qubit_gate)
        if graph is None:
            graph = connectivity_graph(device)

        qubit_chain = random_chain_from_graph(
            graph,
            num_qubits_in_chain,
            seed=seed,
        )

        # Separate the chain into two disjoint layers
        pairwise_chain_indices = [
            (qubit_chain[i], qubit_chain[i + 1]) for i in range(len(qubit_chain) - 1)
        ]
        two_disjoint_layers = [pairwise_chain_indices[0::2], pairwise_chain_indices[1::2]]

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
        if self.params.decompose_clifford_ops:
            circuits = lfexp._transpiled_circuits()
        else:
            circuits = lfexp.circuits()

        return circuits, qubit_chain, two_disjoint_layers, two_qubit_gate, one_qubit_basis_gates

    def dispatch_handler(self, device: "QuantumDevice") -> EPLGData:
        """Generate circuits, submit to device, return EPLGData."""
        circuits, qubit_chain, two_disjoint_layers, two_qubit_gate, one_qubit_basis_gates = (
            self._build_circuits(device)
        )

        shots = self.params.shots
        quantum_job = device.run(circuits, shots=shots)

        # Serialize the two_disjoint_layers as nested lists
        serialized_layers = [[list(edge) for edge in layer] for layer in two_disjoint_layers]

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

        # Deserialize two_disjoint_layers back to tuples
        two_disjoint_layers: list[list[tuple[int, int]]] = [
            [(int(edge[0]), int(edge[1])) for edge in layer]
            for layer in job_data.two_disjoint_layers
        ]

        # Recreate the LayerFidelity experiment to generate circuits and
        # extract metadata
        lfexp = LayerFidelity(
            physical_qubits=job_data.qubit_chain,
            two_qubit_layers=two_disjoint_layers,
            lengths=job_data.lengths,
            num_samples=job_data.num_samples,
            seed=job_data.seed,
            two_qubit_gate=job_data.two_qubit_gate,
            one_qubit_basis_gates=job_data.one_qubit_basis_gates,
        )
        lfexp.experiment_options.max_circuits = 2 * job_data.num_samples * len(job_data.lengths)

        # Optional: bound the RB offset parameter b to stabilize fits on slow-decay devices.
        if getattr(self.params, "constrain_rb_offset_b", False):
            lfexp = tune_layer_fidelity_fits_soft(lfexp, b_half_width=0.05)

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
            backend_name="device",
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

        # Compute EPLG at reference points (10, 20, 50, 100)
        lens_to_eplg = dict(zip(chain_lens, chain_eplgs))
        eplg_10 = lens_to_eplg.get(10, None)
        eplg_20 = lens_to_eplg.get(20, None)
        eplg_50 = lens_to_eplg.get(50, None)
        eplg_100 = lens_to_eplg.get(100, None)

        return EPLGResult(
            chain_lengths=chain_lens,
            chain_eplgs=chain_eplgs,
            eplg_10=eplg_10,
            eplg_20=eplg_20,
            eplg_50=eplg_50,
            eplg_100=eplg_100,
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        """Return circuit batches for resource estimation."""
        circuits, _, _, _, _ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
