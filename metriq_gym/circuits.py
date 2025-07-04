"""Gate-based utility functions."""

import math
import random
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from helpers.lr_qaoa_helpers import PTC_pairs
import networkx as nx
from typing import Literal, List, Tuple

GraphType = Literal[
    "1D", "NL", "FC"
]  # 1D chain of qubits (1D), Native Layout graph (NL), or Fully connected graph (FC).

EncodingType = Literal["Direct", "PTC"]


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


def distribute_edges(graph: nx.Graph) -> list:
    """Distribute edges of a graph into layers, so not two edges in a layer share a node.
       This is needed for optimal circuit depth in a layer of QAOA circuit.
    Args:
        graph: Networkx graph of the problem.
    Returns:
        A list of layers, where each layer contains edges that are connected to nodes of the same degree.
    """
    # Get the maximum degree of the graph
    max_degree = max(dict(graph.degree()).values())
    layers: List[List[Tuple[int, int]]] = [[] for _ in range(max_degree)]
    edges_label = {n: edge for n, edge in enumerate(graph.edges())}
    n_edges = len(edges_label.keys())

    edges_graph = nx.Graph()
    edges_graph.add_nodes_from(edges_label.keys())
    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            # Check if edges share a node and add them to the graph if they do
            if edges_label[i][0] in edges_label[j] or edges_label[i][1] in edges_label[j]:
                edges_graph.add_edges_from([[i, j]])
    # Use greedy coloring to distribute edges into layers
    # Each color represents a layer, where no two edges in the same layer share a node
    colors = nx.greedy_color(edges_graph, strategy="connected_sequential_bfs")
    for node, color in colors.items():
        layers[color].append(list(sorted(edges_label[node])))
    if sum(len(layer) for layer in layers) != n_edges:
        raise ValueError("Not all edges were assigned to a layer, check the graph structure.")
    return layers


def single_cost_layer_circuit(gamma: Parameter, graph: nx.Graph, layers: list) -> QuantumCircuit:
    """Generate a single layer of the QAOA circuit from a networkx graph on a 1D lattice.

    Args:
        gamma: Parameter for the RZZ gate.
        G: Networkx graph of the problem.
    """
    nq = graph.number_of_nodes()  # number of qubits
    qc = QuantumCircuit(nq)
    for layer in layers:
        for i, j in layer:
            qc.rzz(2 * gamma * graph[i][j]["weight"], i, j)
    return qc


def single_cost_layer_PTC_circuit(
    gamma: Parameter, graph: nx.Graph, sequence_2q: list, layer_p_i: int
) -> QuantumCircuit:
    """Generate a single layer of the QAOA circuit from a networkx graph on a PTC (Parity Twine Chain) lattice.
        PTC is the best strategy to create a fully connected graph from a 1D chain of qubits;
       it comes from https://arxiv.org/abs/2408.10907 and https://arxiv.org/abs/2501.14020
        and an implementation for QAOA from https://arxiv.org/abs/2505.17944
    Args:
        gamma: Parameter of the QAOA algorithm.
        graph: Networkx graph of the problem.
        sequence_2q: List of lists, where each inner list contains tuples representing pairs of qubits.
        layer_p_i: Index of the current layer in QAOA .

    Returns:
        QuantumCircuit: A quantum circuit representing the single cost layer.
    """
    nq = graph.number_of_nodes()
    qc = QuantumCircuit(nq)
    depth = len(sequence_2q)
    for k in range(depth):
        for qi, ij in enumerate(sequence_2q[k]):
            if ij in graph.edges() or (ij[0], ij[0]) in graph.edges():
                if k > 0 and ij == sequence_2q[k - 1][qi]:
                    pass
                elif len(ij) == 2:
                    if ij[0] in list(graph[ij[1]].keys()):
                        qc.rz(2 * graph[ij[0]][ij[1]]["weight"] * gamma, qi)
                elif len(ij) == 1:
                    if ij[0] in list(graph[ij[0]].keys()):
                        qc.rz(2 * graph[ij[0]][ij[0]]["weight"] * gamma, qi)
        if k < depth - 1:
            for i in range((k + (depth % 2) * layer_p_i) % 2, nq - 1, 2):
                qc.cx(i + 1, i)
            for j in range(((k + 1) + (depth % 2) * layer_p_i) % 2, nq - 1, 2):
                qc.cx(j, j + 1)
    return qc


def single_mixer_layer_PTC_circuit(
    qc: QuantumCircuit, nq: int, beta: Parameter, sequence_2q: list
) -> None:
    """Generate a single mixer layer of the QAOA circuit using the PTC strategy.
        PTC is the best strategy to create a fully connected graph from a 1D chain of qubits;
        it comes from https://arxiv.org/abs/2408.10907 and https://arxiv.org/abs/2501.14020
        and an implementation for QAOA from https://arxiv.org/abs/2505.17944
    Args:
        qc: QuantumCircuit to which the mixer layer will be added.
        nq: Number of qubits in the circuit.
        beta: Parameter of the QAOA algorithm.
        sequence_2q: List of lists, where each inner list contains tuples representing pairs of qubits.
    """
    qc.rx(-2 * beta, 0)
    last_sequence = sequence_2q[-1]
    for i in range(2, nq, 2):
        for j in last_sequence[i]:
            if j in last_sequence[i - 1]:
                qc.cx(i, i - 1)
                qc.rx(-2 * beta, i)
                qc.cx(i, i - 1)
            elif i < nq - 1 and j not in last_sequence[i + 1]:
                qc.rx(-2 * beta, i)
    for i in range(1, nq, 2):
        for j in last_sequence[i]:
            if j in last_sequence[i - 1]:
                qc.cx(i, i - 1)
                qc.rx(-2 * beta, i)
                qc.cx(i, i - 1)
            elif i < nq - 1 and j not in last_sequence[i + 1]:
                qc.rx(-2 * beta, i)
    qc.rx(-2 * beta, nq - 1)


def single_cost_layer_fully_connected_device(gamma: Parameter, graph: nx.Graph) -> QuantumCircuit:
    """Generate a single layer of the QAOA circuit from a networkx graph with optimal depth in case of
    a fully connected graph in the device.
    Args:
        gamma: Parameter of the QAOA algorithm.
        graph: Networkx graph of the problem.
    Returns:
        QuantumCircuit: A quantum circuit representing the single cost layer.
    """
    nq = graph.number_of_nodes()
    qc = QuantumCircuit(nq)

    permutations = [i for i in range(nq)]

    for qubit_i in range(nq):
        for k in range(qubit_i % 2, nq - 1, 2):
            qubit_pair = (permutations[k], permutations[k + 1])
            qc.rzz(2 * gamma * graph[qubit_pair[0]][qubit_pair[1]]["weight"], *qubit_pair)
            permutations[k], permutations[k + 1] = permutations[k + 1], permutations[k]


def qaoa_circuit(
    graph: nx.Graph, p_layers: int, graph_type: GraphType, circuit_encoding: EncodingType
) -> QuantumCircuit:
    """Generate a QAOA circuit from a networkx graph on a graph_type lattice.

    Args:
        G: Networkx graph of the problem.
        p_layers: Number of QAOA layers.
        graph_type: Type of the graph, "1D":1D chain graph, "NL": Native Layout graph, or "FC": Fully connected graph.
        device_layout_fully_connected: True if the connectivity of the device graph is fully connected.
    Returns:
        QuantumCircuit: A QAOA circuit for the given graph and number of layers.
    """

    gammas = ParameterVector("gamma", length=p_layers)
    betas = ParameterVector("beta", length=p_layers)
    if graph_type == "1D":
        if graph.number_of_edges() != graph.number_of_nodes() - 1:
            raise ValueError(
                "1D graph must be a chain, with number of edges equal to number of nodes - 1."
            )

    nq = graph.number_of_nodes()  # number of qubits
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    if graph_type in ["1D", "NL"]:
        layers = distribute_edges(graph)

        cost_layer = single_cost_layer_circuit(Parameter("theta"), graph, layers)

        for pi in range(p_layers):
            qc.compose(cost_layer.assign_parameters(gammas[pi]), inplace=True)
            qc.rx(-2 * betas[pi], range(nq))

    elif graph_type == "FC":
        if circuit_encoding == "Direct":
            cost_layer = single_cost_layer_fully_connected_device(Parameter("theta"), graph)

            for pi in range(p_layers):
                qc.compose(cost_layer.assign_parameters(gammas[pi]), inplace=True)
                qc.rx(-2 * betas[pi], range(nq))
        elif circuit_encoding == "PTC":
            # If the device is not fully connected use the parity twine chain (PTC) strategy to encode the problem.
            list_parity = PTC_pairs(nq)
            cost_layer_0 = single_cost_layer_PTC_circuit(
                Parameter("theta_0"), graph, list_parity.copy(), 0
            )
            cost_layer_1 = single_cost_layer_PTC_circuit(
                Parameter("theta_1"), graph, list_parity.copy(), 1
            )
            for pi in range(p_layers):
                if pi % 2 == 0:
                    qc.compose(cost_layer_0.assign_parameters(gammas[pi]), inplace=True)
                else:
                    qc.compose(cost_layer_1.assign_parameters(gammas[pi]), inplace=True)
                single_mixer_layer_PTC_circuit(qc, nq, betas[pi], list_parity)
                list_parity = list_parity[::-1]
        else:
            raise ValueError(
                f"{circuit_encoding} is not valid circuit encoding from ['Direct', 'PTC']"
            )
    return qc
