"""Auxiliary utilities for benchmark result post-processing.

Includes functions for calculating statistics from raw measurement results, such as expectation values and uncertainties.
These functions are used in some benchmark implementations to derive metrics (e.g., WIT, QMLKernel)
"""

from math import sqrt
from typing import Mapping

import numpy as np


def effective_shot_count(shots: int, count_results: Mapping[str, int]) -> int:
    total_measurements = sum(count_results.values())
    return total_measurements if total_measurements > 0 else max(shots, 0)


def binary_expectation_value(
    shots: int, count_results: Mapping[str, int], outcome: str = "1"
) -> float:
    effective_shots = effective_shot_count(shots, count_results)
    if effective_shots == 0:
        return 0.0
    return count_results.get(outcome, 0) / effective_shots


def binary_expectation_stddev(
    shots: int, count_results: Mapping[str, int], outcome: str = "1"
) -> float:
    effective_shots = effective_shot_count(shots, count_results)
    if effective_shots == 0:
        return 0.0
    expectation = binary_expectation_value(shots, count_results, outcome=outcome)
    variance = expectation * (1 - expectation) / effective_shots
    return float(sqrt(max(variance, 0.0)))


def _largest_component_from_edges(edges: list[tuple[int, int]], num_nodes: int) -> int:
    """Return the size of the largest connected component for an edge list."""
    if num_nodes <= 0:
        return 0

    parent = list(range(num_nodes))
    sizes = [1] * num_nodes

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(u: int, v: int) -> None:
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return
        if sizes[root_u] < sizes[root_v]:
            root_u, root_v = root_v, root_u
        parent[root_v] = root_u
        sizes[root_u] += sizes[root_v]

    for u, v in edges:
        union(u, v)

    largest = 1
    for idx in range(num_nodes):
        root = find(idx)
        if sizes[root] > largest:
            largest = sizes[root]
    return largest


def bootstrap_largest_component_stddev(
    edge_stats: Mapping[tuple[int, int], tuple[float, float]],
    num_nodes: int,
    *,
    threshold: float,
    num_samples: int = 512,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the uncertainty of the largest connected component via Monte Carlo.

    Each edge ``(u, v)`` is modelled as a Gaussian random variable with mean/std drawn
    from ``edge_stats[(u, v)]``.  During every Monte Carlo draw we sample a synthetic
    value for every edge, mark the edge as ``active`` whenever that draw exceeds the
    violation ``threshold``, and compute the size of the largest connected component
    that can be assembled from the active edges.  The reported uncertainty is the
    sample standard deviation of those largest-component sizes after ``num_samples``
    repetitions.  When all draws produce the same component size the function returns
    zero, indicating that the metric is insensitive to the provided per-edge noise.

    Args:
        edge_stats: Mapping from edge to a tuple of (mean, stddev) metric values (e.g., CHSH scores).
        num_nodes: Total number of nodes in the graph.
        threshold: Threshold that determines whether an edge is considered active.
        num_samples: Number of Monte Carlo samples to draw.
        rng: Optional numpy random generator for deterministic sampling.

    Returns:
        The sample standard deviation of the largest connected component.
    """
    if num_nodes <= 0 or not edge_stats or num_samples <= 1:
        return 0.0

    rng = rng or np.random.default_rng()
    edges = list(edge_stats.items())
    samples = np.empty(num_samples, dtype=float)

    for sample_idx in range(num_samples):
        active_edges: list[tuple[int, int]] = []
        for (u, v), (mean, std) in edges:
            sigma = 0.0 if std is None or np.isnan(std) else float(std)
            value = float(mean) if sigma == 0.0 else float(rng.normal(mean, sigma))
            if value > threshold:
                active_edges.append((u, v))
        samples[sample_idx] = _largest_component_from_edges(active_edges, num_nodes)

    if np.allclose(samples, samples[0]):
        return 0.0
    return float(samples.std(ddof=1))
