"""QUOPS estimator, confidence, and published-result regression tests."""

import math
from statistics import NormalDist

import numpy as np
import pytest

from metriq_gym.benchmarks.quops.analysis import (
    QUOPS_THRESHOLD,
    PolarizationEstimate,
    ShapeEstimate,
    effective_polarization,
    estimate_polarization,
    gated_score,
    normal_lower_bound,
    quops_rate,
)


def test_effective_polarization_ideal_uniform_and_hamming_errors():
    assert effective_polarization({"10": 100}, "10") == 1
    assert effective_polarization({"00": 1, "01": 1, "10": 1, "11": 1}, "10") == 0
    assert effective_polarization({"11": 100}, "10") == pytest.approx(-0.6)
    assert effective_polarization({"01": 100}, "10") == pytest.approx(0.2)
    assert effective_polarization({"1 0": 10}, "10") == 1


@pytest.mark.parametrize(
    "counts,target",
    [
        ({}, "0"),
        ({"0": 0}, "0"),
        ({"0": -1}, "0"),
        ({"0": 1.5}, "0"),
        ({"x": 1}, "0"),
        ({"00": 1}, "0"),
        ({"0": 1}, ""),
        ({"0": 1}, "2"),
    ],
)
def test_effective_polarization_rejects_invalid_data(counts, target):
    with pytest.raises(ValueError):
        effective_polarization(counts, target)


def test_mcfe_normalizes_reference_and_spam_errors():
    # M1=gamma*sqrt(M2*M3), so the recovered target process polarization is .8.
    result = estimate_polarization([0.48] * 100, [0.64] * 100, [0.5625] * 200, seed=3)
    assert result.value == pytest.approx(0.8)
    assert result.standard_error == pytest.approx(0, abs=1e-14)
    assert result.lower_bound == pytest.approx(0.8)
    ideal = estimate_polarization([1] * 10, [1] * 10, [1] * 20, seed=3)
    assert ideal == PolarizationEstimate(1, 0, 1)


def test_paired_circuit_bootstrap_preserves_mixed_reference_covariance():
    # Anti-correlated M1 and M2 have a larger ratio variance than independent draws.
    mixed, reference = np.array([0.9, 0.3] * 50), np.array([0.3, 0.9] * 50)
    estimate = estimate_polarization(mixed, reference, [1] * 200, seed=19)
    independent_variance = (0.09 / 100) / 0.6 + (0.6**2 * 0.09 / 100) / (4 * 0.6**3)
    assert estimate.standard_error > 1.2 * math.sqrt(independent_variance)
    assert estimate.lower_bound == pytest.approx(
        estimate.value - NormalDist().inv_cdf(0.95) * estimate.standard_error
    )
    assert estimate == estimate_polarization(mixed, reference, [1] * 200, seed=19)


@pytest.mark.parametrize(
    "reference,spam",
    [([-0.2] * 4, [1] * 4), ([0] * 4, [1] * 4), ([1] * 4, [-0.2] * 4), ([-0.2] * 4, [-0.2] * 4)],
)
def test_nonpositive_reference_denominators_are_unresolved(reference, spam):
    assert estimate_polarization([0.4] * 4, reference, spam) == PolarizationEstimate(
        None, None, None
    )


def test_invalid_bootstrap_draws_are_not_silently_discarded():
    estimate = estimate_polarization([0.2] * 4, [-1, 1, 1, 1], [1] * 4, seed=0)
    assert estimate.value == pytest.approx(0.2 / math.sqrt(0.5))
    assert estimate.standard_error is None
    assert estimate.lower_bound is None


def test_estimator_does_not_clip_a_sampling_fluctuation_above_one():
    estimate = estimate_polarization([1] * 10, [0.9] * 10, [1] * 10, seed=0)
    assert estimate.value > 1


@pytest.mark.parametrize("kwargs", [{"confidence": 1}, {"bootstrap_samples": 1}])
def test_invalid_estimator_settings(kwargs):
    with pytest.raises(ValueError):
        estimate_polarization([1, 1], [1, 1], [1, 1], **kwargs)


@pytest.mark.parametrize(
    "mixed,reference,spam",
    [([1], [1], [1]), ([1, 1], [1, 1, 1], [1, 1]), ([math.nan, 1], [1, 1], [1, 1])],
)
def test_invalid_ensembles(mixed, reference, spam):
    with pytest.raises(ValueError):
        estimate_polarization(mixed, reference, spam)


def test_gated_score_stops_at_first_failure_even_if_later_shape_passes():
    rows = [
        ShapeEstimate(6, 180, PolarizationEstimate(0.7, 0.01, 0.68)),
        ShapeEstimate(6, 192, PolarizationEstimate(0.62, 0.02, 0.58)),
        ShapeEstimate(6, 204, PolarizationEstimate(0.8, 0.01, 0.78)),
    ]
    assert gated_score(rows) == (180, (6, 180))
    assert gated_score(rows[1:]) == (0, None)
    assert gated_score([]) == (0, None)


def test_unresolved_polarization_cannot_pass():
    assert gated_score([ShapeEstimate(6, 204, PolarizationEstimate(0.9, None, None))]) == (0, None)


@pytest.mark.parametrize("shapes", [[(6, 204), (6, 192)], [(6, 204), (6, 204)], [(4, 128)]])
def test_gated_score_rejects_invalid_predeclared_sequence(shapes):
    with pytest.raises(ValueError):
        gated_score([ShapeEstimate(w, s, PolarizationEstimate(1, 0, 1)) for w, s in shapes])


def test_one_sided_confidence_uses_95_percent_not_two_sided_interval():
    assert normal_lower_bound(0.7, 0.02) == pytest.approx(0.6671029274609705)


def test_boston_published_width_six_gated_score_and_rate():
    # Published width-six in-cone rows, copied from the authors' summary CSV:
    # https://github.com/sandialabs/PRAQTICE/blob/2fb6e9a3a24f4fdf449305d3a915e98073343064/
    # quops/results/data/ibm/ibm_boston/experiment/physical_qubits/1/data.csv
    # This checks analysis of published measurements; it is not a hardware rerun.
    measurements = [
        (48, 0.8750903711599146, 0.007982307591153722),
        (96, 0.7759601550258262, 0.011226222316792628),
        (144, 0.7251803332795622, 0.010899335824014809),
        (156, 0.7059998699257531, 0.013389596266558279),
        (168, 0.7153989003052261, 0.010226702432114662),
        (180, 0.6664388248010766, 0.013488129794841756),
        (192, 0.6579793886873634, 0.011552042489860656),
        (204, 0.6654311115844749, 0.011152167647660272),
        (216, 0.6179421320558182, 0.013256829951227825),
    ]
    rows = [
        ShapeEstimate(6, size, PolarizationEstimate(value, error, normal_lower_bound(value, error)))
        for size, value, error in measurements
    ]
    assert gated_score(rows) == (204, (6, 204))
    assert rows[-1].polarization.value > QUOPS_THRESHOLD
    assert rows[-1].polarization.lower_bound < QUOPS_THRESHOLD
    raw_rate = 705899.2805755396
    gamma = rows[-2].polarization.value
    shots = 200_000  # 100 M1 and 100 M2 circuits, each measured 1,000 times.
    elapsed = 2 * 204 * shots / raw_rate
    assert quops_rate(204, gamma, shots, elapsed) == pytest.approx(312571.18795422756)


def test_quops_rate_accounts_for_mirror_depth_and_polarization():
    assert quops_rate(100, 0.5, 2000, 10) == 10_000
    assert quops_rate(100, 0, 2000, 10) == 0


@pytest.mark.parametrize(
    "arguments", [(100, 0.5, 2000, 0), (100, -0.5, 2000, 1), (100, 0.5, 0, 1), (0, 0.5, 2000, 1)]
)
def test_quops_rate_rejects_invalid_inputs(arguments):
    with pytest.raises(ValueError):
        quops_rate(*arguments)
