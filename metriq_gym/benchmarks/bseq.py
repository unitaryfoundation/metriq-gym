"""BSEQ (Bell state effective qubits) benchmark implementation.

Summary:
    Evaluates how well a device generates Bell pairs that violate the CHSH inequality across
    its connectivity graph. Circuits are built per colouring of the topology and executed in
    four measurement bases to detect correlations.

Connectivity graph:
    The benchmark uses the device's native connectivity graph to determine which qubit pairs
    can be coupled. For superconducting devices (e.g., IBM), this reflects the physical
    coupling map with sparse connectivity. For trapped-ion devices (e.g., IonQ, Quantinuum)
    and simulators, all-to-all connectivity is assumed (complete graph). The graph structure
    affects edge coloring: complete graphs with n qubits require n-1 colors (optimal), while
    sparse topologies typically require fewer colors but test fewer qubit pairs.

Result interpretation:
    Polling returns BSEQResult with:
        - largest_connected_size: size of the biggest connected subgraph of qubit pairs that
          violated CHSH (> 2). Higher means entanglement spans more of the device.
        - fraction_connected: largest_connected_size normalised by the discovered qubit count,
          making it easier to compare devices of different sizes.

References:
    - Original routines attributed to Paul Nation (Qiskit Device Benchmarking).
    - J. F. Clauser et al., Phys. Rev. Lett. 23, 880 (1969).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import networkx as nx
import rustworkx as rx
import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import marginal_counts, sampled_expectation_value

from pydantic import Field
from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.helpers.graph_helpers import (
    GraphColoring,
    device_graph_coloring,
    largest_connected_size,
)
from metriq_gym.qplatform.device import connectivity_graph
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob
    from qbraid.runtime.result_data import MeasCount


class BSEQResult(BenchmarkResult):
    largest_connected_size: int
    fraction_connected: float = Field(...)

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=float(self.largest_connected_size))


@dataclass
class BSEQData(BenchmarkData):
    """Data class to store BSEQ benchmark metadata.

    Attributes:
        shots: Number of shots per quantum circuit execution.
        num_qubits: Number of qubits in the quantum device.
        topology_graph: Graph representing the device topology (optional).
        coloring: Coloring information for circuit partitioning (optional).
    """

    shots: int
    num_qubits: int
    topology_graph: nx.Graph | None = None
    coloring: GraphColoring | dict | None = None


def generate_chsh_circuit_sets(coloring: GraphColoring) -> list[QuantumCircuit]:
    """Generate CHSH circuits based on graph coloring.

    Args:
        coloring: The coloring information of the quantum device topology.

    Returns:
        Nested list of QuantumCircuit objects, grouped by color.
    """
    num_qubits = coloring.num_nodes
    circuits = []
    # For each coloring, generate a set of CHSH pairs (Bell pairs plus an Ry(pi/4)) on each
    # edge of the coloring.  Measurement register is twice the number of qubit pairs in the
    # coloring
    for counter in range(coloring.num_colors):
        num_qubit_pairs = len(
            {key for key, val in coloring.edge_color_map.items() if val == counter}
        )
        qc = QuantumCircuit(num_qubits, 2 * num_qubit_pairs)
        for edge_idx in (key for key, val in coloring.edge_color_map.items() if val == counter):
            edge = (coloring.edge_index_map[edge_idx][0], coloring.edge_index_map[edge_idx][1])
            # For each edge in the color set perform a CHSH experiment at the optimal value
            qc.h(edge[0])
            qc.cx(*edge)
            # Apply CHSH-specific rotation.
            qc.ry(np.pi / 4, edge[0])
        circuits.append(qc)

    exp_sets = []
    # For each coloring circuit, generate 4 new circuits with the required post-rotation operators
    # and measurements appended
    for counter, circ in enumerate(circuits):
        meas_circuits = []
        # Need to create a circuit for each measurement basis. This amounts to appending a H gate to the qubits with an
        # X-basis measurement Each basis corresponds to one of the four CHSH correlation terms.
        for basis in ["ZZ", "ZX", "XZ", "XX"]:
            temp_qc = circ.copy()
            meas_counter = 0
            for edge_idx in (key for key, val in coloring.edge_color_map.items() if val == counter):
                edge = (coloring.edge_index_map[edge_idx][0], coloring.edge_index_map[edge_idx][1])
                for idx, oper in enumerate(basis[::-1]):
                    if oper == "X":
                        temp_qc.h(edge[idx])
                temp_qc.measure(edge, [meas_counter, meas_counter + 1])
                meas_counter += 2
            meas_circuits.append(temp_qc)
        exp_sets.append(meas_circuits)

    return exp_sets


def chsh_subgraph(coloring: GraphColoring, counts: list["MeasCount"]) -> rx.PyGraph:
    """Constructs a subgraph of qubit pairs that violate the CHSH inequality.

    Args:
        job_data: The benchmark metadata including topology and coloring.
        result_data: The result data containing measurement counts.

    Returns:
        The graph of edges that violated the CHSH inequality.
    """
    # A subgraph is constructed containing only the edges (qubit pairs) that successfully violate the CHSH inequality.
    # The size of the largest connected component in this subgraph provides a measure of the device's performance.
    good_edges = []
    for color_idx in range(coloring.num_colors):
        num_meas_pairs = len(
            {key for key, val in coloring.edge_color_map.items() if val == color_idx}
        )
        exp_vals: np.ndarray = np.zeros(num_meas_pairs, dtype=float)

        for idx in range(4):
            for pair in range(num_meas_pairs):
                sub_counts = marginal_counts(counts[color_idx * 4 + idx], [2 * pair, 2 * pair + 1])
                exp_val = sampled_expectation_value(sub_counts, "ZZ")
                exp_vals[pair] += exp_val if idx != 2 else -exp_val

        for idx, edge_idx in enumerate(
            key for key, val in coloring.edge_color_map.items() if val == color_idx
        ):
            edge = (coloring.edge_index_map[edge_idx][0], coloring.edge_index_map[edge_idx][1])
            # The benchmark checks whether the CHSH inequality is violated (i.e., the sum of correlations exceeds 2,
            # indicating entanglement).
            if exp_vals[idx] > 2:
                good_edges.append(edge)

    good_graph = rx.PyGraph(multigraph=False)
    good_graph.add_nodes_from(list(range(coloring.num_nodes)))
    for edge in good_edges:
        good_graph.add_edge(*edge, 1)
    return good_graph


class BSEQ(Benchmark):
    """Benchmark class for BSEQ (Bell state effective qubits) experiments."""

    def _build_circuits(
        self, device: "QuantumDevice"
    ) -> tuple[list[list[QuantumCircuit]], GraphColoring, nx.Graph]:
        """Shared circuit construction logic.

        Args:
            device: The quantum device to build circuits for.

        Returns:
            Tuple of (circuit_sets, coloring, topology_graph).
        """
        topology_graph = connectivity_graph(device)
        coloring = device_graph_coloring(topology_graph)
        circuit_sets = generate_chsh_circuit_sets(coloring)
        return circuit_sets, coloring, topology_graph

    def dispatch_handler(self, device: "QuantumDevice") -> BSEQData:
        """Runs the benchmark and returns job metadata."""
        shots = self.params.shots
        circuit_sets, coloring, topology_graph = self._build_circuits(device)

        quantum_jobs: list[QuantumJob | list[QuantumJob]] = [
            device.run(circ_set, shots=shots) for circ_set in circuit_sets
        ]

        provider_job_ids = [
            job.id
            for quantum_job_set in quantum_jobs
            for job in (quantum_job_set if isinstance(quantum_job_set, list) else [quantum_job_set])
        ]

        return BSEQData(
            provider_job_ids=provider_job_ids,
            shots=shots,
            num_qubits=device.num_qubits,
            topology_graph=topology_graph,
            coloring={
                "num_nodes": coloring.num_nodes,
                "edge_color_map": dict(coloring.edge_color_map),
                "edge_index_map": dict(coloring.edge_index_map),
            },
        )

    def poll_handler(
        self,
        job_data: BSEQData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> BSEQResult:
        """Poll and calculate largest connected component."""
        if not job_data.coloring:
            raise ValueError("Coloring data is required for BSEQ benchmark.")

        coloring = job_data.coloring
        if isinstance(coloring, dict):
            coloring = GraphColoring.from_dict(coloring)
        good_graph = chsh_subgraph(coloring, flatten_counts(result_data))
        lcs = largest_connected_size(good_graph)
        return BSEQResult(
            largest_connected_size=lcs,
            fraction_connected=lcs / coloring.num_nodes,
        )

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list[CircuitBatch]:
        circuit_sets, _, _ = self._build_circuits(device)
        return [
            CircuitBatch(circuits=circuit_group, shots=self.params.shots)
            for circuit_group in circuit_sets
        ]
