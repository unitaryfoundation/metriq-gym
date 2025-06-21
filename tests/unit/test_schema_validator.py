from enum import StrEnum
import json
from unittest.mock import patch
import pytest
from jsonschema import ValidationError
from metriq_gym.schema_validator import load_and_validate

FAKE_BENCHMARK_NAME = "Test Benchmark"

MOCK_SCHEMA_CONTENT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "title": FAKE_BENCHMARK_NAME,
    "properties": {
        "benchmark_name": {"type": "string", "const": FAKE_BENCHMARK_NAME},
        "num_qubits": {"type": "integer", "minimum": 1},
        "shots": {"type": "integer", "minimum": 1},
        "trials": {"type": "integer", "minimum": 1},
    },
    "required": ["benchmark_name", "num_qubits"],
}


class FakeJobType(StrEnum):
    TEST_BENCHMARK = FAKE_BENCHMARK_NAME


@pytest.fixture(autouse=True)
def patch_job_type_enum():
    with patch("metriq_gym.schema_validator.JobType", FakeJobType):
        yield


@pytest.fixture(autouse=True)
def mock_schema(tmpdir):
    schema_file_path = tmpdir.join("test.schema.json")
    with open(schema_file_path, "w") as schema_file:
        json.dump(MOCK_SCHEMA_CONTENT, schema_file)

    SCHEMA_MAPPING = {
        FakeJobType.TEST_BENCHMARK: str(schema_file_path),
    }
    with patch("metriq_gym.schema_validator.SCHEMA_MAPPING", SCHEMA_MAPPING):
        yield


@pytest.fixture
def valid_params():
    return {
        "benchmark_name": FAKE_BENCHMARK_NAME,
        "num_qubits": 5,
        "shots": 1024,
        "trials": 10,
    }


@pytest.fixture
def invalid_params():
    return {
        "benchmark_name": FAKE_BENCHMARK_NAME,
        "num_qubits": 0,  # Invalid value
        "shots": 1024,
        "trials": 10,
    }


@pytest.fixture
def file_path_valid_job(valid_params, tmpdir):
    file_path = tmpdir.join("valid_job.json")
    with open(file_path, "w") as file:
        json.dump(valid_params, file)
    return str(file_path)


@pytest.fixture
def file_path_invalid_job(invalid_params, tmpdir):
    file_path = tmpdir.join("invalid_job.json")
    with open(file_path, "w") as file:
        json.dump(invalid_params, file)
    return str(file_path)


def test_load_and_validate_valid(file_path_valid_job, valid_params):
    params_model = load_and_validate(file_path_valid_job)
    assert params_model.model_dump() == valid_params


def test_load_and_validate_invalid(file_path_invalid_job, mock_schema):
    with pytest.raises(ValidationError):
        load_and_validate(file_path_invalid_job)


def test_load_and_validate_invalid_job_path():
    with pytest.raises(FileNotFoundError):
        load_and_validate("invalid_job.json")
