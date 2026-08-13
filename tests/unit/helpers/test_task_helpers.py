import pytest
from metriq_gym.helpers.task_helpers import flatten_counts
from qbraid.runtime.result_data import MeasCount, GateModelResultData


@pytest.fixture
def ibm_result_data():
    return [
        GateModelResultData(
            measurement_counts=[MeasCount({"00": 50, "11": 50}), MeasCount({"00": 30, "11": 70})]
        )
    ]


@pytest.fixture
def aws_result_data():
    return [
        GateModelResultData(measurement_counts=MeasCount({"00": 50, "11": 50})),
        GateModelResultData(measurement_counts=MeasCount({"00": 30, "11": 70})),
    ]


@pytest.fixture
def mixed_result_data():
    return [
        GateModelResultData(
            measurement_counts=[MeasCount({"00": 50, "11": 50}), MeasCount({"00": 30, "11": 70})]
        ),
        GateModelResultData(measurement_counts=MeasCount({"00": 20, "11": 80})),
    ]


def test_flatten_counts_ibm(ibm_result_data):
    flat_counts = flatten_counts(ibm_result_data)
    assert len(flat_counts) == 2
    assert flat_counts[0] == MeasCount({"00": 50, "11": 50})
    assert flat_counts[1] == MeasCount({"00": 30, "11": 70})


def test_flatten_counts_aws(aws_result_data):
    flat_counts = flatten_counts(aws_result_data)
    assert len(flat_counts) == 2
    assert flat_counts[0] == MeasCount({"00": 50, "11": 50})
    assert flat_counts[1] == MeasCount({"00": 30, "11": 70})


def test_flatten_counts_mixed(mixed_result_data):
    flat_counts = flatten_counts(mixed_result_data)
    assert len(flat_counts) == 3
    assert flat_counts[0] == MeasCount({"00": 50, "11": 50})
    assert flat_counts[1] == MeasCount({"00": 30, "11": 70})
    assert flat_counts[2] == MeasCount({"00": 20, "11": 80})


def test_flatten_counts_empty():
    flat_counts = flatten_counts([])
    assert flat_counts == []


def test_flatten_counts_none_raises():
    result_data = [GateModelResultData(measurement_counts=None)]
    with pytest.raises(ValueError, match="no measurement counts or probabilities"):
        flatten_counts(result_data)


def test_flatten_counts_probabilities_only_raises():
    """Regression test for #799: a provider (e.g. OriginQ simulators) that returns
    probabilities instead of sampled counts must fail loudly, not silently drop the
    result and let the benchmark report a fabricated 0.0 score."""
    result_data = [
        GateModelResultData(measurement_counts=None, measurement_probabilities={"1001": 1.0})
    ]
    with pytest.raises(ValueError, match="returned measurement probabilities instead"):
        flatten_counts(result_data)


def test_flatten_counts_partial_none_raises():
    """One good result and one result with no counts must still raise, not silently
    return only the good result's counts."""
    result_data = [
        GateModelResultData(measurement_counts=MeasCount({"00": 50, "11": 50})),
        GateModelResultData(measurement_counts=None),
    ]
    with pytest.raises(ValueError):
        flatten_counts(result_data)
