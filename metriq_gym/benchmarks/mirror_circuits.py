"""
Mirror circuits benchmark for the Metriq Gym.

This benchmark evaluates a quantum device's ability to execute mirror circuits,
which are quantum circuits with a reflection structure that perform calculations
and then reverse them. Mirror circuits provide scalable benchmarking capabilities
for quantum computers as defined in Proctor et al., arXiv:2008.11294.
"""

from dataclasses import dataclass
from enum import StrEnum

import rustworkx as rx
import numpy as np
from rustworkx.generators import path_graph
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, CZGate
from qiskit.quantum_info import random_clifford, random_pauli
from qiskit.quantum_info import Clifford, Pauli
from numpy import random
from typing import Sequence

from pydantic import Field
from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
    MetricDirection,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.qplatform.device import connectivity_graph

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


class TwoQubitGateType(StrEnum):
    CNOT = "CNOT"
    CZ = "CZ"


class MirrorCircuitsResult(BenchmarkResult):
    success_probability: BenchmarkScore = Field(..., json_schema_extra={"direction": MetricDirection.HIGHER})
    polarization: BenchmarkScore = Field(..., json_schema_extra={"direction": MetricDirection.HIGHER})
    binary_success: bool


@dataclass
class MirrorCircuitsData(BenchmarkData):
    num_layers: int
    two_qubit_gate_prob: float
    two_qubit_gate_name: str
    shots: int
    num_qubits: int
    num_circuits: int
    seed: int | None
    expected_bitstrings: list[str]


POLARIZATION_THRESHOLD = 1 / np.e


def select_optimal_qubit_subset(topology_graph: rx.PyGraph, target_width: int) -> list[int]:
    """
    Select a connected subset of qubits with the best connectivity.

    Args:
        topology_graph: The device connectivity graph.
        target_width: Number of qubits to select.

    Returns:
        List of selected qubit indices.
    """
    all_qubits = list(topology_graph.node_indices())

    if target_width >= len(all_qubits):
        return all_qubits

    if target_width == 1:
        return [all_qubits[0]]

    # Use a greedy approach similar to CLOPS create_qubit_map
    # Start from the qubit with highest degree and expand
    degrees = [(node, topology_graph.degree(node)) for node in all_qubits]
    degrees.sort(key=lambda x: x[1], reverse=True)

    selected = [degrees[0][0]]  # Start with highest degree node
    remaining = set(all_qubits) - {degrees[0][0]}

    # Greedily add connected qubits
    while len(selected) < target_width and remaining:
        best_candidate = None
        max_connections = -1

        for candidate in remaining:
            # Count connections to already selected qubits
            connections = sum(
                1
                for selected_qubit in selected
                if topology_graph.has_edge(candidate, selected_qubit)
            )
            if connections > max_connections:
                max_connections = connections
                best_candidate = candidate

        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            # No more connected qubits, add any remaining
            selected.append(list(remaining)[0])
            remaining.remove(selected[-1])

    return selected[:target_width]


def create_subgraph_from_qubits(
    topology_graph: rx.PyGraph, selected_qubits: list[int]
) -> rx.PyGraph:
    """
    Create a subgraph containing only the selected qubits and their connections.

    Args:
        topology_graph: Original device topology graph.
        selected_qubits: List of qubit indices to include.

    Returns:
        Subgraph with selected qubits and their interconnections.
    """
    subgraph = rx.PyGraph()

    # Add nodes with remapped indices
    subgraph.add_nodes_from(range(len(selected_qubits)))

    # Add edges between selected qubits
    for i, qubit_i in enumerate(selected_qubits):
        for j, qubit_j in enumerate(selected_qubits):
            if i < j and topology_graph.has_edge(qubit_i, qubit_j):
                subgraph.add_edge(i, j, None)

    return subgraph


def working_graph(width: int) -> rx.PyGraph:
    """Return the connectivity graph used to synthesize mirror circuits."""

    return path_graph(width)


def random_paulis(
    connectivity_graph: rx.PyGraph, random_state: random.RandomState
) -> QuantumCircuit:
    """Returns a circuit with randomly selected Pauli gates on each qubit.

    Args:
        connectivity_graph: Connectivity graph of device to run circuit on.
        random_state: Random state to select Paulis I, X, Y, Z uniformly at random.

    Returns:
        A quantum circuit with random Pauli gates on each qubit.
    """
    num_qubits = len(connectivity_graph.node_indices())
    if num_qubits == 0:
        return QuantumCircuit(0)

    qc = QuantumCircuit(num_qubits)

    for qubit in connectivity_graph.node_indices():
        pauli = random_pauli(1, group_phase=False, seed=random_state.randint(2**31))
        pauli_instruction = pauli.to_instruction()
        qc.append(pauli_instruction, [qubit])

    return qc


def random_single_cliffords(
    connectivity_graph: rx.PyGraph, random_state: random.RandomState, base_seed: int | None = None
) -> QuantumCircuit:
    """Returns a circuit with randomly selected single-qubit Clifford gates on each qubit.

    This function generates a uniformly random single-qubit Clifford from the full 24-element
    single-qubit Clifford group for each qubit, rather than just sampling from generators.
    This ensures proper uniform sampling over all elements of the group, addressing the
    statistical requirements for robust randomized benchmarking.

    Args:
        connectivity_graph: Connectivity graph of device to run circuit on.
        random_state: Random state to select Cliffords uniformly at random from the full group.
        base_seed: Base seed for deterministic generation. If provided, derives seeds deterministically.

    Returns:
        A quantum circuit with random single-qubit Clifford gates on each qubit.
    """
    num_qubits = len(connectivity_graph.node_indices())
    if num_qubits == 0:
        return QuantumCircuit(0)

    qc = QuantumCircuit(num_qubits)

    for i, qubit in enumerate(connectivity_graph.node_indices()):
        clifford_seed = (base_seed + i) if base_seed is not None else None

        clifford = random_clifford(1, seed=clifford_seed)
        clifford_circuit = clifford.to_circuit()

        if clifford_circuit.num_qubits > 0:
            for instruction in clifford_circuit.data:
                gate = instruction.operation
                qc.append(gate, [qubit])

    return qc


def edge_grab(
    two_qubit_gate_prob: float,
    connectivity_graph: rx.PyGraph,
    random_state: random.RandomState,
) -> rx.PyGraph:
    """Selects edges for two-qubit gates based on probability and connectivity.

    This function implements a greedy edge selection algorithm that ensures
    no qubit participates in more than one two-qubit gate simultaneously,
    respecting hardware constraints while providing controllable gate density.

    Args:
        two_qubit_gate_prob: Probability of an edge being chosen from candidates.
        connectivity_graph: The connectivity graph for the backend.
        random_state: Random state to select edges uniformly at random.

    Returns:
        A graph with selected edges for two-qubit gates.
    """
    connectivity_graph = connectivity_graph.copy()
    candidate_edges = rx.PyGraph()
    final_edges = rx.PyGraph()

    candidate_edges.add_nodes_from(connectivity_graph.node_indices())
    final_edges.add_nodes_from(connectivity_graph.node_indices())

    while connectivity_graph.edge_list():
        edges = connectivity_graph.edge_list()
        if not edges:
            break
        num = random_state.randint(len(edges))
        curr_edge = edges[num]
        candidate_edges.add_edge(curr_edge[0], curr_edge[1], None)

        nodes_to_remove = [curr_edge[0], curr_edge[1]]
        for node in nodes_to_remove:
            if node in connectivity_graph.node_indices():
                connectivity_graph.remove_node(node)

    for edge in candidate_edges.edge_list():
        if random_state.uniform(0.0, 1.0) < two_qubit_gate_prob:
            final_edges.add_edge(edge[0], edge[1], None)

    return final_edges


def random_cliffords(
    connectivity_graph: rx.PyGraph,
    random_state: random.RandomState,
    two_qubit_gate_name: str = "CNOT",
    base_seed: int | None = None,
) -> QuantumCircuit:
    """Generate random Clifford gates on selected edges.

    Applies the specified two-qubit gate to all edges in the connectivity graph,
    and applies uniformly random single-qubit Cliffords to all isolated qubits.
    This maintains the Clifford structure required for efficient classical simulation
    while providing comprehensive randomization.

    Args:
        connectivity_graph: Graph with edges for two-qubit gates.
        random_state: Random state to choose Cliffords uniformly at random.
        two_qubit_gate_name: Two-qubit gate to use ("CNOT" or "CZ").
        base_seed: Base seed for deterministic generation.

    Returns:
        A circuit with two-qubit gates and random single-qubit Cliffords.

    Raises:
        ValueError: If two_qubit_gate_name is not supported.
    """
    if len(connectivity_graph.node_indices()) == 0:
        return QuantumCircuit(0)

    try:
        gate_type = TwoQubitGateType(two_qubit_gate_name)
    except ValueError:
        raise ValueError(
            f"Unsupported two-qubit gate: {two_qubit_gate_name}. "
            f"Supported gates: {list(TwoQubitGateType)}"
        )

    num_qubits = max(connectivity_graph.node_indices()) + 1
    qc = QuantumCircuit(num_qubits)

    two_qubit_gate = CXGate() if gate_type == TwoQubitGateType.CNOT else CZGate()
    for edge in connectivity_graph.edge_list():
        qc.append(two_qubit_gate, [edge[0], edge[1]])

    nodes_with_edges = set()
    for edge in connectivity_graph.edge_list():
        nodes_with_edges.add(edge[0])
        nodes_with_edges.add(edge[1])

    isolated_qubits = [
        node for node in connectivity_graph.node_indices() if node not in nodes_with_edges
    ]

    for i, qubit in enumerate(isolated_qubits):
        clifford_seed = (base_seed + i + 1000) if base_seed is not None else None

        clifford = random_clifford(1, seed=clifford_seed)
        clifford_circuit = clifford.to_circuit()

        if clifford_circuit.num_qubits > 0:
            for instruction in clifford_circuit.data:
                gate = instruction.operation
                qc.append(gate, [qubit])

    return qc
    
def pauli_from_layer(pauli_layer: QuantumCircuit) -> Pauli:
    """
    Convert a "middle_pauli" layer (only I/X/Y/Z per qubit) to a qiskit Pauli.
    """
    n = pauli_layer.num_qubits
    per_qubit = ["I"] * n  # default all identity

    for instr, qargs, _ in pauli_layer.data:
        name = instr.name.lower()
        if name in ("barrier", "delay", "measure"):
            continue  # middle layer shouldn't have these, but be permissive
        if len(qargs) != 1:
            raise ValueError(f"Non-1q op '{instr.name}' found in middle_pauli layer.")
        # Get the circuit's index for this qubit (portable across Terra versions)
        q = pauli_layer.find_bit(qargs[0]).index
        if name in ("x", "y", "z"):
            per_qubit[q] = name.upper()
        elif name in ("id", "i"):
            per_qubit[q] = "I"
        else:
            raise ValueError(f"Non-Pauli op '{instr.name}' found in middle_pauli layer.")

    label = ''.join(per_qubit[::-1])
    return Pauli(label)
    
def expected_bitstring_without_simulation(
    initial_clifford_layer: QuantumCircuit,
    forward_layers: Sequence[QuantumCircuit],
    middle_pauli: QuantumCircuit,
) -> str:
    n = initial_clifford_layer.num_qubits

    fwd = initial_clifford_layer.copy()
    for lyr in forward_layers:
        fwd.compose(lyr, inplace=True)
    c_fwd = Clifford(fwd)

    P_mid = pauli_from_layer(middle_pauli)
    P_conj = P_mid.evolve(c_fwd)

    bits_little_endian = ["1" if P_conj.x[i] else "0" for i in range(n)]
    # Qiskit counts use MSB-left bitstring formatting, so reverse the little-endian bits.
    return ''.join(bits_little_endian[::-1])

def assert_forward_is_clifford(initial_clifford_layer, forward_layers):
    fwd = initial_clifford_layer.copy()
    for lyr in forward_layers:
        fwd.compose(lyr, inplace=True)
    # Raises if the circuit contains non-Clifford ops/angles
    Clifford(fwd)
    
def generate_mirror_circuit(
    num_layers: int,
    two_qubit_gate_prob: float,
    connectivity_graph: rx.PyGraph,
    two_qubit_gate_name: str = "CNOT",
    seed: int | None = None,
) -> tuple[QuantumCircuit, str]:
    """Generate a mirror circuit with specified parameters.

    Creates a quantum circuit with a mirror structure: forward layers of random
    Clifford operations, a middle layer of random Paulis, then the inverse of
    the forward layers. The expected outcome is computed by ideal simulation,
    ensuring accurate benchmarking against the theoretical prediction.

    Args:
        num_layers: The number of random Clifford layers to be generated.
        two_qubit_gate_prob: Probability of a two-qubit gate being applied (0.0 to 1.0).
        connectivity_graph: The connectivity graph of the backend.
        two_qubit_gate_name: Name of two-qubit gate to use ("CNOT" or "CZ").
        seed: Seed for generating randomized mirror circuit.

    Returns:
        A tuple of (mirror_circuit, expected_bitstring) where expected_bitstring
        is computed from noiseless simulation of the circuit.

    Raises:
        ValueError: If parameters are invalid.
    """
    if not 0 <= two_qubit_gate_prob <= 1:
        raise ValueError("two_qubit_gate_prob must be between 0 and 1")

    try:
        TwoQubitGateType(two_qubit_gate_name)
    except ValueError:
        raise ValueError(f"two_qubit_gate_name must be one of {list(TwoQubitGateType)}")

    random_state = random.RandomState(seed)
    num_qubits = len(connectivity_graph.node_indices())

    if num_qubits == 0:
        qc = QuantumCircuit(1)
        qc.measure_all()
        return qc, "0"

    qc = QuantumCircuit(num_qubits)

    initial_clifford_layer = random_single_cliffords(connectivity_graph, random_state, seed)
    qc.compose(initial_clifford_layer, inplace=True)
    qc.barrier()

    forward_layers = []
    for layer_idx in range(num_layers):
        pauli_layer = random_paulis(connectivity_graph, random_state)
        qc.compose(pauli_layer, inplace=True)
        qc.barrier()
        forward_layers.append(pauli_layer)

        selected_edges = edge_grab(two_qubit_gate_prob, connectivity_graph, random_state)
        layer_seed = seed + layer_idx if seed is not None else None
        clifford_layer = random_cliffords(
            selected_edges, random_state, two_qubit_gate_name, layer_seed
        )
        qc.compose(clifford_layer, inplace=True)
        qc.barrier()
        forward_layers.append(clifford_layer)
    
    assert_forward_is_clifford(initial_clifford_layer, forward_layers)
    
    middle_pauli = random_paulis(connectivity_graph, random_state)
    qc.compose(middle_pauli, inplace=True)
    qc.barrier()

    for layer in reversed(forward_layers):
        qc.compose(layer.inverse(), inplace=True)
        qc.barrier()

    qc.compose(initial_clifford_layer.inverse(), inplace=True)
    qc.barrier()

    qc.measure_all()
    
    sim_circuit = qc.copy()
    sim_circuit.remove_final_measurements()

    expected_bitstring = expected_bitstring_without_simulation(
        initial_clifford_layer=initial_clifford_layer,
        forward_layers=forward_layers,
        middle_pauli=middle_pauli
    )

    return qc, expected_bitstring


class MirrorCircuits(Benchmark):
    def dispatch_handler(self, device: "QuantumDevice") -> MirrorCircuitsData:
        num_layers = self.params.num_layers
        two_qubit_gate_prob = self.params.two_qubit_gate_prob
        two_qubit_gate_name = self.params.two_qubit_gate_name
        shots = self.params.shots
        num_circuits = self.params.num_circuits
        seed = self.params.seed
        target_width = getattr(self.params, "width", None)
        if not isinstance(target_width, (int, type(None))):
            target_width = None
        topology_graph = connectivity_graph(device)

        available_qubits = len(topology_graph.node_indices())

        if available_qubits < 2:
            raise ValueError("Mirror circuits benchmark requires a device with at least two qubits")

        # Select subset of qubits if width is specified
        if target_width is not None:
            if target_width < 2:
                raise ValueError(
                    f"Requested width {target_width} is too small; mirror circuits require at least two qubits"
                )
            if target_width > available_qubits:
                raise ValueError(
                    f"Requested width {target_width} exceeds device capacity {available_qubits}"
                )

            actual_width = target_width
        else:
            actual_width = available_qubits

        working_graph_conn = working_graph(actual_width)

        circuits = []
        expected_bitstrings = []

        for i in range(num_circuits):
            circuit_seed = None if seed is None else seed + i
            circuit, expected_bitstring = generate_mirror_circuit(
                num_layers=num_layers,
                two_qubit_gate_prob=two_qubit_gate_prob,
                connectivity_graph=working_graph_conn,
                two_qubit_gate_name=two_qubit_gate_name,
                seed=circuit_seed,
            )
            circuits.append(circuit)
            expected_bitstrings.append(expected_bitstring)

        return MirrorCircuitsData.from_quantum_job(
            quantum_job=device.run(circuits, shots=shots),
            num_layers=num_layers,
            two_qubit_gate_prob=two_qubit_gate_prob,
            two_qubit_gate_name=two_qubit_gate_name,
            shots=shots,
            num_qubits=actual_width,
            num_circuits=num_circuits,
            seed=seed,
            expected_bitstrings=expected_bitstrings,
        )

    def poll_handler(
        self,
        job_data: MirrorCircuitsData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> MirrorCircuitsResult:
        counts_list = flatten_counts(result_data)

        if job_data.num_qubits == 0:
            raise ValueError("Mirror circuits benchmark requires at least 1 qubit")

        # --- Pool counts across circuits ---
        successes_total = 0
        shots_total = 0
        for counts, expected_bitstring in zip(counts_list, job_data.expected_bitstrings):
            shots = sum(counts.values())
            shots_total += shots
            successes_total += counts.get(expected_bitstring, 0)
        if shots_total == 0:
            final_success_probability = 0.0
            final_success_prob_err = 0.0
        else:
            s = successes_total / shots_total
            final_success_probability = s
            final_success_prob_err = np.sqrt(s * (1.0 - s) / shots_total)  # binomial SE

        w = job_data.num_qubits
        baseline = 1.0 / (2**w)
        polarization = (
            (final_success_probability - baseline) / (1.0 - baseline)
            if (1.0 - baseline) > 0
            else 0.0
        )
        polarization_err = final_success_prob_err / (1.0 - baseline) if polarization > 0 else 0.0

        return MirrorCircuitsResult(
            success_probability=BenchmarkScore(
                value=final_success_probability,
                uncertainty=final_success_prob_err,
            ),
            polarization=BenchmarkScore(
                value=polarization,
                uncertainty=polarization_err,
            ),
            binary_success=bool(polarization >= POLARIZATION_THRESHOLD),
        )