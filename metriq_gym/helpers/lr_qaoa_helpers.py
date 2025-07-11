import numpy as np
import networkx as nx
import math
from docplex.mp.model import Model


def weighted_maxcut_solver(graph: nx.Graph) -> str:
    """Constructs a mathematical model for the Max-Cut problem using the CPLEX solver.
    This function creates a binary optimization model to maximize the cut in a weighted graph.
    Args:
        graph (networkx.Graph): The input weighted graph where edges represent cut costs.
    Returns:
        str: A binary string representing the optimal partition of the graph nodes (e.g., "1010").
    """
    mdl = Model("MaxCut")
    num_nodes = graph.number_of_nodes()
    x = {i: mdl.binary_var(name=f"x_{i}") for i in range(num_nodes)}
    mdl.minimize(
        mdl.sum(graph[i][j]["weight"] * (2 * x[i] * x[j] - x[i] - x[j]) for (i, j) in graph.edges)
    )
    mdl.solve()
    optimal_solution = "".join(
        str(round(mdl.solution.get_value(var))) for var in mdl.iter_binary_vars()
    )
    return optimal_solution


def cost_maxcut(bitstring: str, graph: nx.Graph) -> float:
    """
    Computes the cost of a given bitstring solution for the Max-Cut problem.

    Parameters:
    bitstring (str): A binary string representing a partition of the graph nodes (e.g., "1010").
    graph (networkx.Graph): The input weighted graph where edges represent cut costs.

    Returns:
    float: The computed cost of the Max-Cut solution.
    """
    cost = 0
    for i, j in graph.edges():
        if bitstring[i] + bitstring[j] in ["10", "01"]:
            cost += graph[i][j]["weight"]
    return cost


def objective_func(samples_dict: dict, graph: nx.Graph, optimal: str) -> dict:
    """
    Evaluates the performance of LR-QAOA for the Max-Cut problem.

    Parameters:
    samples_dict (dict): A dictionary where keys are bitstrings (binary solutions),
                         and values are their occurrence counts.
    graph (networkx.Graph): The input weighted graph where edges represent cut costs.
    optimal (str): The optimal bitstring solution found by classical solvers (e.g., CPLEX).

    Returns:
    dict: A dictionary containing:
        - "r": The expected approximation ratio.
        - "probability": The probability of sampling the optimal solution.
    """

    max_cost = cost_maxcut(optimal, graph)
    probability = 0.0
    total_cost = 0.0
    shots = 0.0
    for bitstring, counts in samples_dict.items():
        cost = cost_maxcut(bitstring, graph)
        total_cost += counts * cost
        if math.isclose(cost, max_cost):
            probability += counts

        if cost > max_cost:
            print(f"There is a better cost than that of CPLEX: {cost - max_cost}")
        shots += counts
    r = total_cost / (max_cost * shots)
    probability /= shots
    return {"r": r, "probability": probability}


def random_samples(num_samples: int, n_qubits: int) -> dict:
    """
    Generates random bitstring samples for a given number of qubits.

    Parameters:
    num_samples (int): The number of random bitstrings to generate.
    n_qubits (int): The number of qubits (length of each bitstring).

    Returns:
    dict: A dictionary where keys are randomly generated bitstrings
          and values are their occurrence counts.
    """

    random_samples = {}

    for _ in range(num_samples):
        bitstring = "".join(str(i) for i in np.random.choice([0, 1], n_qubits))
        if bitstring not in random_samples:
            random_samples[bitstring] = 0
        random_samples[bitstring] += 1

    return random_samples


def SWAP_pairs(nq: int) -> list:
    """Generate pairs of qubits for a 2-qubit gate in a SWAP configuration.
       SWAP gates between neighboring pairs in a brickwork pattern.
    Args:
        nq: Number of qubits in the circuit.
    Returns:
        A list of lists, where each inner list contains tuples representing pairs of qubits."""
    qubit_order = list(range(nq))
    list_2q = [[(qubit_order[ii], qubit_order[ii + 1]) for ii in range(0, nq - 1, 2)]]
    for i in range(1, nq):
        for j in range(i % 2, nq - 1, 2):
            qubit_order[j], qubit_order[j + 1] = qubit_order[j + 1], qubit_order[j]
        list_2q.append([(qubit_order[ii], qubit_order[ii + 1]) for ii in range(i % 2, nq - 1, 2)])
    return list_2q
