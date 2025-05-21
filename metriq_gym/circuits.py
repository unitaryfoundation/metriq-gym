"""Gate-based utility functions."""

import math
import random
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
import networkx as nx
import numpy as np


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

def single_1D_layer_circuit(gamma:Parameter, G: nx.Graph) -> QuantumCircuit:
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

def qaoa_1D_circuit(G:nx.Graph, p:int):
    """Generate a QAOA circuit from a networkx graph on a 1D lattice.

    Args:
        G: Networkx graph of the problem.
        p: Number of layers.
    """
    nq = G.number_of_nodes() # number of qubits
    max_weight = max(abs([G[i][j]["weight"] for i, j in G.edges()]))
    gammas = ParameterVector("gamma", length=p)
    betas = ParameterVector("beta", length=p)
    
    single_layer = single_1D_layer_circuit(Parameter("theta"), G)
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    for pi in range(p):
        qc.compose(single_layer.assign_parameters(gammas[pi]/max_weight), inplace=True)
        qc.rx(-2*betas[pi], range(nq))
    return qc
