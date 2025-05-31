"""
Mirror circuits benchmark for the Metriq Gym.

This benchmark evaluates a quantum device's ability to execute mirror circuits,
which are quantum circuits with a reflection structure that perform calculations
and then reverse them. Mirror circuits provide scalable benchmarking capabilities
for quantum computers as defined in Proctor et al., Nature Physics 2022.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import networkx as nx
import numpy as np
from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, YGate, ZGate, IGate, HGate, SGate, TGate
from qiskit.circuit.library import CXGate, CZGate
from numpy import random

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts, flatten_job_ids
from metriq_gym.qplatform.device import connectivity_graph


class MirrorCircuitsResult(BenchmarkResult):
    """Results from mirror circuits benchmark execution."""
    success_probability: float
    polarization: float
    binary_success: bool


@dataclass
class MirrorCircuitsData(BenchmarkData):
    """Data class to store mirror circuits benchmark metadata."""
    nlayers: int
    two_qubit_gate_prob: float
    two_qubit_gate_name: str
    shots: int
    num_qubits: int
    seed: Optional[int]
    expected_bitstring: str


def random_paulis(connectivity_graph: nx.Graph, random_state: random.RandomState) -> QuantumCircuit:
    """Returns a circuit with randomly selected Pauli gates on each qubit.

    Args:
        connectivity_graph: Connectivity graph of device to run circuit on.
        random_state: Random state to select Paulis I, X, Y, Z uniformly at random.
    """
    num_qubits = len(connectivity_graph.nodes)
    qc = QuantumCircuit(num_qubits)
    
    paulis = [IGate(), XGate(), YGate(), ZGate()]
    
    for qubit in connectivity_graph.nodes:
        pauli_gate = paulis[random_state.randint(len(paulis))]
        qc.append(pauli_gate, [qubit])
    
    return qc


def edge_grab(
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    random_state: random.RandomState,
) -> nx.Graph:
    """Selects edges for two-qubit gates based on probability and connectivity.
    
    Args:
        two_qubit_gate_prob: Probability of an edge being chosen.
        connectivity_graph: The connectivity graph for the backend.
        random_state: Random state to select edges uniformly at random.

    Returns:
        A graph with selected edges for two-qubit gates.
    """
    connectivity_graph = connectivity_graph.copy()
    candidate_edges = nx.Graph()
    final_edges = nx.Graph()
    final_edges.add_nodes_from(connectivity_graph.nodes)

    while connectivity_graph.edges:
        edges = list(connectivity_graph.edges)
        if not edges:
            break
        num = random_state.randint(len(edges))
        curr_edge = edges[num]
        candidate_edges.add_edge(*curr_edge)
        connectivity_graph.remove_nodes_from(curr_edge)

    for edge in candidate_edges.edges:
        if random_state.uniform(0.0, 1.0) < two_qubit_gate_prob:
            final_edges.add_edge(*edge)
    
    return final_edges


def random_cliffords(
    connectivity_graph: nx.Graph,
    random_state: random.RandomState,
    two_qubit_gate_name: str = "CNOT",
) -> QuantumCircuit:
    """Generate random Clifford gates on selected edges.
    
    Args:
        connectivity_graph: Graph with edges for two-qubit gates.
        random_state: Random state to choose Cliffords uniformly at random.
        two_qubit_gate_name: Two-qubit gate to use ("CNOT" or "CZ").

    Returns:
        A circuit with two-qubit gates and random single-qubit Cliffords.
    """
    if not connectivity_graph.nodes:
        return QuantumCircuit(0)
    
    num_qubits = max(connectivity_graph.nodes) + 1
    qc = QuantumCircuit(num_qubits)
    
    # Add two-qubit gates
    two_qubit_gate = CXGate() if two_qubit_gate_name == "CNOT" else CZGate()
    for edge in connectivity_graph.edges:
        qc.append(two_qubit_gate, [edge[0], edge[1]])
    
    # Add random single-qubit Cliffords to isolated qubits
    isolated_qubits = list(nx.isolates(connectivity_graph))
    single_cliffords = [IGate(), XGate(), YGate(), ZGate(), HGate(), SGate()]
    
    for qubit in isolated_qubits:
        clifford = single_cliffords[random_state.randint(len(single_cliffords))]
        qc.append(clifford, [qubit])
    
    return qc


def generate_mirror_circuit(
    nlayers: int,
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    two_qubit_gate_name: str = "CNOT",
    seed: Optional[int] = None,
) -> Tuple[QuantumCircuit, str]:
    """Generate a mirror circuit with specified parameters.
    
    Args:
        nlayers: The number of random Clifford layers to be generated.
        two_qubit_gate_prob: Probability of a two-qubit gate being applied.
        connectivity_graph: The connectivity graph of the backend.
        two_qubit_gate_name: Name of two-qubit gate to use ("CNOT" or "CZ").
        seed: Seed for generating randomized mirror circuit.

    Returns:
        A tuple of (mirror_circuit, expected_bitstring).
    """
    if not 0 <= two_qubit_gate_prob <= 1:
        raise ValueError("two_qubit_gate_prob must be between 0 and 1")

    if two_qubit_gate_name not in ["CNOT", "CZ"]:
        raise ValueError("two_qubit_gate_name must be 'CNOT' or 'CZ'")

    random_state = random.RandomState(seed)
    num_qubits = len(connectivity_graph.nodes)
    
    if num_qubits == 0:
        qc = QuantumCircuit(1)
        qc.measure_all()
        return qc, "0"

    # Initialize circuit
    qc = QuantumCircuit(num_qubits)
    
    # Forward circuit layers
    forward_layers = []
    for _ in range(nlayers):
        # Random Pauli layer
        pauli_layer = random_paulis(connectivity_graph, random_state)
        qc = qc.compose(pauli_layer)
        forward_layers.append(pauli_layer)
        
        # Random Clifford layer with two-qubit gates
        selected_edges = edge_grab(two_qubit_gate_prob, connectivity_graph, random_state)
        clifford_layer = random_cliffords(selected_edges, random_state, two_qubit_gate_name)
        qc = qc.compose(clifford_layer)
        forward_layers.append(clifford_layer)
    
    # Middle random Pauli layer
    middle_pauli = random_paulis(connectivity_graph, random_state)
    qc = qc.compose(middle_pauli)
    
    # Reverse (mirror) the forward layers
    for layer in reversed(forward_layers):
        qc = qc.compose(layer.inverse())
    
    # Add measurements
    qc.measure_all()
    
    # Expected bitstring is all zeros for a perfect mirror circuit
    expected_bitstring = "0" * num_qubits
    
    return qc, expected_bitstring


class MirrorCircuits(Benchmark):
    """Benchmark class for mirror circuits experiments."""

    def dispatch_handler(self, device: QuantumDevice) -> MirrorCircuitsData:
        """Runs the benchmark and returns job metadata."""
        nlayers = self.params.nlayers
        two_qubit_gate_prob = self.params.two_qubit_gate_prob
        two_qubit_gate_name = self.params.two_qubit_gate_name
        shots = self.params.shots
        seed = getattr(self.params, 'seed', None)

        topology_graph = connectivity_graph(device)
        
        # Convert rustworkx graph to networkx for compatibility
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(topology_graph.node_indices())
        nx_graph.add_edges_from(topology_graph.edge_list())

        circuit, expected_bitstring = generate_mirror_circuit(
            nlayers=nlayers,
            two_qubit_gate_prob=two_qubit_gate_prob,
            connectivity_graph=nx_graph,
            two_qubit_gate_name=two_qubit_gate_name,
            seed=seed
        )

        quantum_job = device.run(circuit, shots=shots)
        provider_job_ids = flatten_job_ids(quantum_job)

        return MirrorCircuitsData(
            provider_job_ids=provider_job_ids,
            nlayers=nlayers,
            two_qubit_gate_prob=two_qubit_gate_prob,
            two_qubit_gate_name=two_qubit_gate_name,
            shots=shots,
            num_qubits=device.num_qubits or len(nx_graph.nodes),
            seed=seed,
            expected_bitstring=expected_bitstring
        )

    def poll_handler(
        self,
        job_data: MirrorCircuitsData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> MirrorCircuitsResult:
        """Poll and calculate mirror circuit success metrics."""
        counts = flatten_counts(result_data)[0]
        
        # Calculate success probability
        total_shots = sum(counts.values())
        success_count = counts.get(job_data.expected_bitstring, 0)
        success_probability = success_count / total_shots if total_shots > 0 else 0.0
        
        # Calculate polarization P = (S - 1/2^w) / (1 - 1/2^w)
        w = job_data.num_qubits
        if w > 0:
            baseline = 1.0 / (2 ** w)
            polarization = (success_probability - baseline) / (1.0 - baseline) if (1.0 - baseline) > 0 else 0.0
        else:
            polarization = success_probability
        
        # Binary success (typically threshold is 2/3)
        binary_success = success_probability >= (2.0 / 3.0)
        
        return MirrorCircuitsResult(
            success_probability=success_probability,
            polarization=polarization,
            binary_success=binary_success
        )
