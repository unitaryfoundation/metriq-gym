from dataclasses import dataclass
import logging
import pytest
from unittest.mock import MagicMock, patch

from qbraid import QbraidError
from qbraid.runtime import JobStatus
from metriq_gym.benchmarks.benchmark import BenchmarkData, BenchmarkResult
from metriq_gym.run import (
    setup_device,
    dispatch_job,
    fetch_result,
)
from metriq_gym.job_manager import MetriqGymJob, JobManager
from metriq_gym.constants import JobType
from metriq_gym.exceptions import QBraidSetupError


class FakeDevice:
    def __init__(self, id):
        self.id = id


@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def mock_device():
    return FakeDevice(id="test_device")


@pytest.fixture
def patch_load_provider(mock_provider, monkeypatch):
    monkeypatch.setattr("metriq_gym.run.load_provider", lambda _: mock_provider)


@pytest.fixture
def mock_args():
    """Create mock args for testing dispatch functions."""
    args = MagicMock()
    args.provider = "test_provider"
    args.device = "test_device"
    args.benchmark_configs = ["test.json"]
    return args


@pytest.fixture
def mock_job_manager():
    """Create mock job manager for testing."""
    job_manager = MagicMock()
    job_manager.add_job.return_value = "test-job-id-123"
    return job_manager


def test_setup_device_success(mock_provider, mock_device, patch_load_provider):
    mock_provider.get_device.return_value = mock_device

    provider_name = "test_provider"
    backend_name = "test_backend"

    device = setup_device(provider_name, backend_name)

    mock_provider.get_device.assert_called_once_with(backend_name)
    assert device == mock_device


@patch("metriq_gym.run.get_providers")
def test_setup_device_invalid_provider(get_providers_patch, caplog):
    get_providers_patch.return_value = ["supported_provider"]
    caplog.set_level(logging.INFO)

    provider_name = "unsupported_provider"
    backend_name = "whatever_backend"

    with pytest.raises(QBraidSetupError, match="Provider not found"):
        setup_device(provider_name, backend_name)

    # Verify the printed output
    assert f"No provider matching the name '{provider_name}' found." in caplog.text
    assert "Providers available: ['supported_provider']" in caplog.text


def test_setup_device_invalid_device(mock_provider, patch_load_provider, caplog):
    caplog.set_level(logging.INFO)
    mock_provider.get_device.side_effect = QbraidError()
    mock_provider.get_devices.return_value = [FakeDevice(id="device1"), FakeDevice(id="device2")]

    provider_name = "test_provider"
    backend_name = "non_existent_backend"

    with pytest.raises(QBraidSetupError, match="Device not found"):
        setup_device(provider_name, backend_name)

    # Verify the printed output
    assert (
        f"No device matching the name '{backend_name}' found in provider '{provider_name}'."
        in caplog.text
    )
    assert "Devices available: ['device1', 'device2']" in caplog.text


@patch("os.path.exists")
def test_dispatch_missing_config_file(mock_exists, mock_args, mock_job_manager, capsys):
    """Test behavior when configuration file is missing."""
    # Setup mocks
    mock_args.benchmark_config = "missing.json"
    mock_exists.return_value = False  # Simulate missing files

    # Mock setup_device to avoid provider validation error
    with patch("metriq_gym.run.setup_device") as mock_setup_device:
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device

        # Execute function
        dispatch_job(mock_args, mock_job_manager)

        # Verify output shows file not found errors
        captured = capsys.readouterr()
        assert "Configuration file not found" in captured.out


class DummyResult(BenchmarkResult):
    value: int


@dataclass
class DummyJobData(BenchmarkData):
    provider_job_ids: list[str]


class DummyQuantumJob:
    def __init__(self, job_id: str, value: int):
        self.id = job_id
        self._value = value

    def status(self):
        return JobStatus.COMPLETED

    def result(self):
        class _R:
            data = {"value": self._value}

        return _R()


class DummyBenchmark:
    def poll_handler(self, job_data, result_data, quantum_jobs):
        return DummyResult(value=result_data[0]["value"])


def _make_cached_job(val: int) -> MetriqGymJob:
    return MetriqGymJob(
        id="job-1",
        job_type=JobType.WIT,
        params={"benchmark_name": JobType.WIT.name},
        data={"provider_job_ids": ["provider-job-1"]},
        provider_name="local",
        device_name="dummy_device",
        platform={},
        dispatch_time=None,
        result_data={"value": val},
    )


def test_fetch_result_uses_cache_when_no_flag(monkeypatch):
    EXPECTED_CACHED_VALUE = 7
    job = _make_cached_job(EXPECTED_CACHED_VALUE)
    jm = JobManager()
    jm.jobs.append(job)
    args = MagicMock()
    args.no_cache = False

    import metriq_gym.run as run_mod

    monkeypatch.setattr(run_mod, "setup_benchmark_result_class", lambda *_: DummyResult)
    monkeypatch.setattr(run_mod, "setup_job_data_class", lambda *_: DummyJobData)
    monkeypatch.setattr(run_mod, "setup_benchmark", lambda *_, **__: DummyBenchmark())
    monkeypatch.setattr(
        run_mod,
        "load_job",
        lambda *_, **__: DummyQuantumJob("provider-job-1", EXPECTED_CACHED_VALUE),
    )
    monkeypatch.setattr(run_mod, "validate_and_create_model", lambda params: params)

    result = fetch_result(job, args, jm)
    assert result.value == EXPECTED_CACHED_VALUE


def test_fetch_result_bypasses_cache_with_flag(monkeypatch):
    EXPECTED_FRESH_VALUE = 42
    CACHED_VALUE = 7
    job = _make_cached_job(CACHED_VALUE)
    jm = JobManager()
    jm.jobs.append(job)
    args = MagicMock()
    args.no_cache = True

    import metriq_gym.run as run_mod

    monkeypatch.setattr(run_mod, "setup_benchmark_result_class", lambda *_: DummyResult)
    monkeypatch.setattr(run_mod, "setup_job_data_class", lambda *_: DummyJobData)
    monkeypatch.setattr(run_mod, "setup_benchmark", lambda *_, **__: DummyBenchmark())
    monkeypatch.setattr(
        run_mod,
        "load_job",
        lambda *_, **__: DummyQuantumJob("provider-job-1", EXPECTED_FRESH_VALUE),
    )
    monkeypatch.setattr(run_mod, "validate_and_create_model", lambda params: params)

    result = fetch_result(job, args, jm)
    assert result.value == EXPECTED_FRESH_VALUE, (
        "Should fetch fresh value when --no-cache specified"
    )
    assert job.result_data == {"value": EXPECTED_FRESH_VALUE}, (
        "Cached result_data should be updated"
    )
