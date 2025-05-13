import numpy as np
from collections import defaultdict
import networkx as nx

def cost_maxcut(bitstring: str, G: nx.Graph) -> float:
    """
    Computes the cost of a given bitstring solution for the Max-Cut problem.

    Parameters:
    bitstring (str): A binary string representing a partition of the graph nodes (e.g., "1010").
    weights (dict): A dictionary where keys are edge tuples (i, j) and values are edge weights.

    Returns:
    float: The computed cost of the Max-Cut solution.
    """
    cost = 0  # Initialize the cost
    
    # Iterate through all edges in the graph
    for i, j in G.edges():
        # Check if the nodes i and j are in different partitions (cut condition)
        if bitstring[i] + bitstring[j] in ["10", "01"]:
            cost += G[i][j]["weight"]  # Add the edge weight to the cost
    return cost  # Return the total cut cost

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

    # Compute the cost of the optimal Max-Cut solution
    max_cost = cost_maxcut(optimal, G)

    # Iterate through all sampled bitstrings
    probability = 0 
    total_cost = 0
    shots = 0
    for bitstring, counts in samples_dict.items():
        cost = cost_maxcut(bitstring, G)  # Compute cost of the given bitstring
        total_cost += counts * cost         
        # If this bitstring matches the optimal cost, update probability
        if abs(cost - max_cost) < 1e-6:
            probability += counts
        
        # Check if a better-than-optimal solution appears (sanity check)
        if cost > max_cost:
            print(f"There is a better cost than that of CPLEX: {cost - max_cost}")
        shots += counts  # Update total shots
 
    # Compute the expected approximation ratio
    r = total_cost / (max_cost * shots)

    # Normalize the probability of sampling the optimal solution
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
    
    random_samples = defaultdict(int)  # Dictionary to store bitstrings and their counts

    # Generate random bitstrings and count their occurrences
    for _ in range(num_samples):
        bitstring = "".join(str(i) for i in np.random.choice([0, 1], n_qubits))  # Generate a random bitstring
        random_samples[bitstring] += 1  # Increment count for the generated bitstring

    return random_samples