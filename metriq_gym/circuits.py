"""Gate-based utility functions."""

import math
import random
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
import networkx as nx
import numpy as np
from typing import Literal

def rand_u3(circ: QuantumCircuit, q: int) -> None:
    """Apply a random U3 gate to a specified qubit in the given quantum circuit.

    Args:
        circ: QuantumCircuit instance representing the circuit.
        q: The qubit index in the circuit where the U3 gate will be applied.
    """
    th = random.uniform(0, 2 * math.pi)
    ph = random.uniform(0, 2 * math.pi)
    lm = random.uniform(0, 2 * math.pi)
    circ.u(th, ph, lm, q)


def qiskit_random_circuit_sampling(n: int) -> QuantumCircuit:
    """Generate a square circuit, for random circuit sampling

    Args:
        n: Width of circuit to generate.
    """
    circ = QuantumCircuit(n)
    for _ in range(n):
        for i in range(n):
            rand_u3(circ, i)

        unused_bits = list(range(n))
        random.shuffle(unused_bits)
        while len(unused_bits) > 1:
            c = unused_bits.pop()
            t = unused_bits.pop()
            circ.cx(c, t)
    return circ

def single_1d_layer_circuit(gamma:Parameter, G: nx.Graph) -> QuantumCircuit:
    """Generate a single layer of the QAOA circuit from a networkx graph on a 1D lattice.

    Args:
        gamma: Parameter for the RZZ gate.
        G: Networkx graph of the problem.
    """
    nq = G.number_of_nodes() # number of qubits
    qc = QuantumCircuit(nq)
    layers = [[(i,i+1) for i in range(j, nq-1, 2)] for j in range(2)]
    for layer in layers:
        for (i,j) in layer:
            qc.rzz(2 * gamma * G[i][j]["weight"], i, j)
    return qc

def qaoa_1d_circuit(gammas:ParameterVector, betas:ParameterVector, max_weight:float, G: nx.Graph, p_layers: int) -> QuantumCircuit:
    """Generate a QAOA circuit from a networkx graph on a 1D lattice.

    Args:
        G: Networkx graph of the problem.
        p_layers: Number of QAOA layers.
    """
    nq = G.number_of_nodes() # number of qubits
    number_of_edges = G.number_of_edges()

    if number_of_edges != nq - 1:
        raise ValueError(f"Graph is not a 1D chain, it has {number_of_edges} edges but should have {nq - 1} edges.")
    
    single_layer = single_1d_layer_circuit(Parameter("theta"), G)
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    for pi in range(p_layers):
        qc.compose(single_layer.assign_parameters(gammas[pi]/max_weight), inplace=True)
        qc.rx(-2*betas[pi], range(nq))
    return qc

GraphType = Literal["1D", "NL", "FC"] # 1D chain of qubits (1D), Native Layout graph (NL), or Fully connected graph (FC).

def qaoa_circuit(G:nx.Graph, p_layers:int, graph_type:GraphType) -> QuantumCircuit:
    """Generate a QAOA circuit from a networkx graph on a graph_type lattice.

    Args:
        G: Networkx graph of the problem.
        p_layers: Number of QAOA layers.
        graph_type: Type of the graph, "1D":1D chain graph, "NL": Native Layout graph, or "FC": Fully connected graph.
    """
    
    max_weight = max(abs([G[i][j]["weight"] for i, j in G.edges()]))
    gammas = ParameterVector("gamma", length=p_layers)
    betas = ParameterVector("beta", length=p_layers)
    if graph_type == "1D":
        qc = qaoa_1d_circuit(gammas, betas, max_weight, G, p_layers)
    
    return qc
