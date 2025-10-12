import numpy as np

from metriq_gym.benchmarks.bseq import (
    CHSH_THRESHOLD,
    estimate_lcs_uncertainty,
)


def test_estimate_lcs_uncertainty_returns_zero_without_edges():
    assert estimate_lcs_uncertainty({}, num_nodes=4) == 0.0


def test_estimate_lcs_uncertainty_zero_variance_is_deterministic():
    edge_stats = {(0, 1): (CHSH_THRESHOLD + 0.5, 0.0)}
    rng = np.random.default_rng(0)
    assert estimate_lcs_uncertainty(edge_stats, num_nodes=3, rng=rng) == 0.0


def test_estimate_lcs_uncertainty_reflects_sampling_noise():
    edge_stats = {(0, 1): (CHSH_THRESHOLD + 0.1, 0.3)}
    rng = np.random.default_rng(1234)
    std = estimate_lcs_uncertainty(edge_stats, num_nodes=3, rng=rng, num_samples=256)
    assert 0.0 < std < 3.0
