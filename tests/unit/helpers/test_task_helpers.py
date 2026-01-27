import pytest
from metriq_gym.helpers.task_helpers import flatten_counts, serialize_raw_counts
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


def test_flatten_counts_none():
    result_data = [GateModelResultData(measurement_counts=None)]
    flat_counts = flatten_counts(result_data)
    assert flat_counts == []


# Tests for serialize_raw_counts


def test_serialize_raw_counts_ibm(ibm_result_data):
    """Test serializing IBM-style result data (list of counts in one result)."""
    serialized = serialize_raw_counts(ibm_result_data)
    assert len(serialized) == 1
    assert serialized[0]["measurement_counts"] == [
        {"00": 50, "11": 50},
        {"00": 30, "11": 70},
    ]


def test_serialize_raw_counts_aws(aws_result_data):
    """Test serializing AWS-style result data (single counts per result)."""
    serialized = serialize_raw_counts(aws_result_data)
    assert len(serialized) == 2
    assert serialized[0]["measurement_counts"] == {"00": 50, "11": 50}
    assert serialized[1]["measurement_counts"] == {"00": 30, "11": 70}


def test_serialize_raw_counts_mixed(mixed_result_data):
    """Test serializing mixed result data."""
    serialized = serialize_raw_counts(mixed_result_data)
    assert len(serialized) == 2
    assert serialized[0]["measurement_counts"] == [
        {"00": 50, "11": 50},
        {"00": 30, "11": 70},
    ]
    assert serialized[1]["measurement_counts"] == {"00": 20, "11": 80}


def test_serialize_raw_counts_empty():
    """Test serializing empty result data."""
    serialized = serialize_raw_counts([])
    assert serialized == []


def test_serialize_raw_counts_none():
    """Test serializing result data with None counts."""
    result_data = [GateModelResultData(measurement_counts=None)]
    serialized = serialize_raw_counts(result_data)
    assert len(serialized) == 1
    assert serialized[0]["measurement_counts"] is None
