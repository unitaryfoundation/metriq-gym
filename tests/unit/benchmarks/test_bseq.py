import numpy as np
import pytest

from metriq_gym.benchmarks.bseq import BSEQResult, CHSH_THRESHOLD
from metriq_gym.benchmarks.benchmark import BenchmarkScore
from metriq_gym.helpers.statistics import bootstrap_largest_component_stddev


def test_bootstrap_stddev_returns_zero_without_edges():
    assert bootstrap_largest_component_stddev({}, num_nodes=4, threshold=CHSH_THRESHOLD) == 0.0


def test_bootstrap_stddev_zero_variance_is_deterministic():
    edge_stats = {(0, 1): (CHSH_THRESHOLD + 0.5, 0.0)}
    rng = np.random.default_rng(0)
    assert (
        bootstrap_largest_component_stddev(
            edge_stats, num_nodes=3, threshold=CHSH_THRESHOLD, rng=rng
        )
        == 0.0
    )


def test_bootstrap_stddev_reflects_sampling_noise():
    edge_stats = {(0, 1): (CHSH_THRESHOLD + 0.1, 0.3)}
    rng = np.random.default_rng(1234)
    std = bootstrap_largest_component_stddev(
        edge_stats,
        num_nodes=3,
        threshold=CHSH_THRESHOLD,
        rng=rng,
        num_samples=256,
    )
    assert 0.0 < std < 3.0


def test_bseq_result_exposes_benchmark_score():
    result = BSEQResult(largest_connected_size=BenchmarkScore(value=8.0, uncertainty=1.5))
    assert result.values == pytest.approx({"largest_connected_size": 8.0})
    assert result.uncertainties == pytest.approx({"largest_connected_size": 1.5})
