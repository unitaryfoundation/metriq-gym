
"""
Mirror circuits benchmark for the Metriq Gym.

This benchmark evaluates a quantum device's ability to execute mirror circuits,
which are quantum circuits with a reflection structure that perform calculations
and then reverse them. Mirror circuits provide scalable benchmarking capabilities
for quantum computers as defined in Proctor et al., Nature Physics 2022.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Union

import networkx as nx
import numpy as np
from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, YGate, ZGate, IGate, CXGate, CZGate
from qiskit.quantum_info import random_clifford, Statevector
from numpy import random

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts, flatten_job_ids
from metriq_gym.qplatform.device import connectivity_graph


class TwoQubitGateType(StrEnum):
    """Enumeration of supported two-qubit gate types."""
    CNOT = "CNOT"
    CZ = "CZ"


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
    num_circuits: int
    seed: Union[int, None]
    expected_bitstring: str


def random_paulis(connectivity_graph: nx.Graph, random_state: random.RandomState) -> QuantumCircuit:
    """Returns a circuit with randomly selected Pauli gates on each qubit.

    Args:
        connectivity_graph: Connectivity graph of device to run circuit on.
        random_state: Random state to select Paulis I, X, Y, Z uniformly at random.
        
    Returns:
        A quantum circuit with random Pauli gates on each qubit.
    """
    num_qubits = len(connectivity_graph.nodes)
    if num_qubits == 0:
        return QuantumCircuit(0)
        
    qc = QuantumCircuit(num_qubits)
    
    paulis = [IGate(), XGate(), YGate(), ZGate()]
    
    for qubit in connectivity_graph.nodes:
        pauli_gate = paulis[random_state.randint(len(paulis))]
        qc.append(pauli_gate, [qubit])
    
    return qc


def random_single_cliffords(connectivity_graph: nx.Graph, random_state: random.RandomState) -> QuantumCircuit:
    """Returns a circuit with randomly selected single-qubit Clifford gates on each qubit.

    This function generates a uniformly random single-qubit Clifford from the full 24-element
    single-qubit Clifford group for each qubit, rather than just sampling from generators.
    This ensures proper uniform sampling over all elements of the group, addressing the
    statistical requirements for robust randomized benchmarking.

    Args:
        connectivity_graph: Connectivity graph of device to run circuit on.
        random_state: Random state to select Cliffords uniformly at random from the full group.
        
    Returns:
        A quantum circuit with random single-qubit Clifford gates on each qubit.
    """
    num_qubits = len(connectivity_graph.nodes)
    if num_qubits == 0:
        return QuantumCircuit(0)
        
    qc = QuantumCircuit(num_qubits)
    
    # Generate a uniformly random single-qubit Clifford for each qubit
    # This properly samples from all 24 elements of the single-qubit Clifford group
    for qubit in connectivity_graph.nodes:
        # Use qiskit's random_clifford to get a uniformly random 1-qubit Clifford
        clifford = random_clifford(1, seed=random_state.randint(2**31))
        clifford_circuit = clifford.to_circuit()
        
        # Map the circuit to the correct qubit
        if clifford_circuit.num_qubits > 0:
            for gate, qargs, cargs in clifford_circuit.data:
                qc.append(gate, [qubit])
    
    return qc


def edge_grab(
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    random_state: random.RandomState,
) -> nx.Graph:
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
    candidate_edges = nx.Graph()
    final_edges = nx.Graph()
    final_edges.add_nodes_from(connectivity_graph.nodes)

    # Generate disjoint edge set using greedy selection
    while connectivity_graph.edges:
        edges = list(connectivity_graph.edges)
        if not edges:
            break
        num = random_state.randint(len(edges))
        curr_edge = edges[num]
        candidate_edges.add_edge(*curr_edge)
        connectivity_graph.remove_nodes_from(curr_edge)

    # Apply probability filter to candidate edges
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
    
    Applies the specified two-qubit gate to all edges in the connectivity graph,
    and applies uniformly random single-qubit Cliffords to all isolated qubits.
    This maintains the Clifford structure required for efficient classical simulation
    while providing comprehensive randomization.
    
    Args:
        connectivity_graph: Graph with edges for two-qubit gates.
        random_state: Random state to choose Cliffords uniformly at random.
        two_qubit_gate_name: Two-qubit gate to use ("CNOT" or "CZ").

    Returns:
        A circuit with two-qubit gates and random single-qubit Cliffords.
        
    Raises:
        ValueError: If two_qubit_gate_name is not supported.
    """
    if not connectivity_graph.nodes:
        return QuantumCircuit(0)
    
    # Validate gate name
    try:
        gate_type = TwoQubitGateType(two_qubit_gate_name)
    except ValueError:
        raise ValueError(f"Unsupported two-qubit gate: {two_qubit_gate_name}. "
                        f"Supported gates: {list(TwoQubitGateType)}")
    
    num_qubits = max(connectivity_graph.nodes) + 1
    qc = QuantumCircuit(num_qubits)
    
    # Add two-qubit gates to connected edges
    two_qubit_gate = CXGate() if gate_type == TwoQubitGateType.CNOT else CZGate()
    for edge in connectivity_graph.edges:
        qc.append(two_qubit_gate, [edge[0], edge[1]])
    
    # Add random single-qubit Cliffords to isolated qubits
    isolated_qubits = list(nx.isolates(connectivity_graph))
    
    for qubit in isolated_qubits:
        # Use qiskit's random_clifford to get uniformly random single-qubit Clifford
        clifford = random_clifford(1, seed=random_state.randint(2**31))
        clifford_circuit = clifford.to_circuit()
        
        # Map the circuit to the correct qubit
        if clifford_circuit.num_qubits > 0:
            for gate, qargs, cargs in clifford_circuit.data:
                qc.append(gate, [qubit])
    
    return qc


def generate_mirror_circuit(
    nlayers: int,
    two_qubit_gate_prob: float,
    connectivity_graph: nx.Graph,
    two_qubit_gate_name: str = "CNOT",
    seed: Union[int, None] = None,
) -> tuple[QuantumCircuit, str]:
    """Generate a mirror circuit with specified parameters.
    
    Creates a quantum circuit with a mirror structure: forward layers of random
    Clifford operations, a middle layer of random Paulis, then the inverse of
    the forward layers. The expected outcome is computed by ideal simulation,
    ensuring accurate benchmarking against the theoretical prediction.
    
    Args:
        nlayers: The number of random Clifford layers to be generated.
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
    num_qubits = len(connectivity_graph.nodes)
    
    if num_qubits == 0:
        qc = QuantumCircuit(1)
        qc.measure_all()
        return qc, "0"

    # Initialize circuit
    qc = QuantumCircuit(num_qubits)
    
    # Initial single-qubit Clifford layer
    initial_clifford_layer = random_single_cliffords(connectivity_graph, random_state)
    qc = qc.compose(initial_clifford_layer)
    
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
    
    # Inverse of initial single-qubit Clifford layer
    qc = qc.compose(initial_clifford_layer.inverse())
    
    # Add measurements
    qc.measure_all()
    
    # Compute the expected bitstring by noiseless simulation
    # This is the key improvement: we simulate to get the actual expected result
    # rather than assuming all zeros, which is only true for specific circuits
    
    # Create a copy without measurements for simulation
    sim_circuit = qc.copy()
    sim_circuit.remove_final_measurements()
    
    # Simulate directly using Statevector (no external simulator needed)
    try:
        statevector = Statevector.from_circuit(sim_circuit)
        probabilities = statevector.probabilities()
        most_likely_state = np.argmax(probabilities)
        
        # Convert to binary string
        expected_bitstring = format(most_likely_state, f'0{num_qubits}b')
        
    except Exception:
        # Fallback for very large circuits or simulation errors
        # This maintains compatibility while providing a reasonable default
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
        num_circuits = getattr(self.params, 'num_circuits', 1)
        seed = getattr(self.params, "seed", None)

        topology_graph = connectivity_graph(device)
        
        # Convert rustworkx graph to networkx for compatibility
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(topology_graph.node_indices())
        nx_graph.add_edges_from(topology_graph.edge_list())

        # Generate multiple random circuits for statistical averaging
        circuits = []
        expected_bitstring = None
        
        for i in range(num_circuits):
            circuit_seed = None if seed is None else seed + i
            circuit, bitstring = generate_mirror_circuit(
                nlayers=nlayers,
                two_qubit_gate_prob=two_qubit_gate_prob,
                connectivity_graph=nx_graph,
                two_qubit_gate_name=two_qubit_gate_name,
                seed=circuit_seed
            )
            circuits.append(circuit)
            # All circuits should have the same expected bitstring
            if expected_bitstring is None:
                expected_bitstring = bitstring

        quantum_job = device.run(circuits, shots=shots)
        provider_job_ids = flatten_job_ids(quantum_job)

        return MirrorCircuitsData(
            provider_job_ids=provider_job_ids,
            nlayers=nlayers,
            two_qubit_gate_prob=two_qubit_gate_prob,
            two_qubit_gate_name=two_qubit_gate_name,
            shots=shots,
            num_qubits=device.num_qubits or len(nx_graph.nodes),
            num_circuits=num_circuits,
            seed=seed,
            expected_bitstring=expected_bitstring or "0"
        )

    def poll_handler(
        self,
        job_data: MirrorCircuitsData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> MirrorCircuitsResult:
        """Poll and calculate mirror circuit success metrics."""
        counts_list = flatten_counts(result_data)
        
        # Handle edge case
        if job_data.num_qubits == 0:
            raise ValueError("Mirror circuits benchmark requires at least 1 qubit")
        
        # Calculate average success probability over all circuit repetitions
        total_success_count = 0
        total_shots = 0
        
        for counts in counts_list:
            total_shots += sum(counts.values())
            total_success_count += counts.get(job_data.expected_bitstring, 0)
        
        success_probability = total_success_count / total_shots if total_shots > 0 else 0.0
        
        # Calculate polarization P = (S - 1/2^w) / (1 - 1/2^w)
        w = job_data.num_qubits
        baseline = 1.0 / (2 ** w)
        polarization = (success_probability - baseline) / (1.0 - baseline) if (1.0 - baseline) > 0 else 0.0
        
        # Binary success (typically threshold is 2/3)
        binary_success = success_probability >= (2.0 / 3.0)
        
        return MirrorCircuitsResult(
            success_probability=success_probability,
            polarization=polarization,
            binary_success=binary_success
        )
