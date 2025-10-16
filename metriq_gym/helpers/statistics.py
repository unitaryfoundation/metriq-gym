"""Helper utilities for benchmark result post-processing."""

from math import sqrt
from typing import Mapping


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
