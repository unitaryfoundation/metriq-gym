import numpy as np
from collections import defaultdict
import networkx as nx
import math

def cost_maxcut(bitstring: str, G: nx.Graph) -> float:
    """
    Computes the cost of a given bitstring solution for the Max-Cut problem.

    Parameters:
    bitstring (str): A binary string representing a partition of the graph nodes (e.g., "1010").
    G (networkx.Graph): The input weighted graph where edges represent cut costs.

    Returns:
    float: The computed cost of the Max-Cut solution.
    """
    cost = 0
    for i, j in G.edges():
        if bitstring[i] + bitstring[j] in ["10", "01"]:
            cost += G[i][j]["weight"] 
    return cost

def objective_func(samples_dict: dict, G: nx.Graph, optimal: str) -> dict:
    """
    Evaluates the performance of LR-QAOA for the Max-Cut problem.

    Parameters:
    samples_dict (dict): A dictionary where keys are bitstrings (binary solutions), 
                         and values are their occurrence counts.
    G (networkx.Graph): The input weighted graph where edges represent cut costs.
    optimal (str): The optimal bitstring solution found by classical solvers (e.g., CPLEX).

    Returns:
    dict: A dictionary containing:
        - "r": The expected approximation ratio.
        - "probability": The probability of sampling the optimal solution.
    """

    max_cost = cost_maxcut(optimal, G)
    probability = 0 
    total_cost = 0
    shots = 0
    for bitstring, counts in samples_dict.items():
        cost = cost_maxcut(bitstring, G) 
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
    
    random_samples = defaultdict(int)

    for _ in range(num_samples):
        bitstring = "".join(str(i) for i in np.random.choice([0, 1], n_qubits))
        random_samples[bitstring] += 1

    return random_samples