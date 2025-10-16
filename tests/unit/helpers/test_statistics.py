import pytest

from metriq_gym.helpers.statistics import (
    binary_expectation_stddev,
    binary_expectation_value,
    effective_shot_count,
)


def test_effective_shot_count_prefers_recorded_counts():
    counts = {"0": 40, "1": 60}
    assert effective_shot_count(10, counts) == 100


def test_effective_shot_count_falls_back_to_requested_shots():
    assert effective_shot_count(256, {}) == 256
    assert effective_shot_count(0, {}) == 0


def test_binary_expectation_value_uses_default_outcome_one():
    counts = {"0": 3, "1": 7}
    assert binary_expectation_value(10, counts) == pytest.approx(0.7)


def test_binary_expectation_value_accepts_custom_outcome():
    counts = {"0": 9}
    # Only nine total measurements were taken; outcome "0" should report 1.0.
    assert binary_expectation_value(100, counts, outcome="0") == pytest.approx(1.0)
    # Outcome "1" is missing, so expectation should drop to 0.0.
    assert binary_expectation_value(100, counts, outcome="1") == 0.0


def test_binary_expectation_stddev_matches_binomial_formula():
    counts = {"0": 40, "1": 60}
    stddev = binary_expectation_stddev(100, counts)
    assert stddev == pytest.approx(0.04898979485566356)


def test_binary_expectation_stddev_handles_zero_counts():
    assert binary_expectation_stddev(100, {}) == 0.0
