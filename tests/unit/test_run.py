from dataclasses import dataclass
from datetime import datetime
import logging
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import BaseModel
from qbraid import QbraidError
from qbraid.runtime import JobStatus
from metriq_gym.benchmarks.benchmark import BenchmarkData, BenchmarkResult, BenchmarkScore
from metriq_gym.run import (
    setup_device,
    dispatch_job,
    fetch_result,
    estimate_job,
)
from metriq_gym.job_manager import MetriqGymJob, JobManager
from metriq_gym.constants import JobType
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.resource_estimation import (
    GateCounts,
    CircuitEstimate,
    ResourceEstimate,
    quantinuum_hqc_formula,
)


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
    assert "Providers available: supported_provider" in caplog.text


@patch("metriq_gym.run.get_providers")
def test_setup_device_empty_provider(get_providers_patch, caplog):
    get_providers_patch.return_value = ["supported_provider"]
    caplog.set_level(logging.INFO)

    provider_name = ""
    backend_name = "whatever_backend"

    with pytest.raises(QBraidSetupError, match="Provider not found"):
        setup_device(provider_name, backend_name)

    assert "No provider name specified." in caplog.text
    assert "Providers available: supported_provider" in caplog.text


def test_setup_device_invalid_device(mock_provider, patch_load_provider, caplog):
    caplog.set_level(logging.INFO)
    mock_provider.get_device.side_effect = QbraidError()
    mock_provider.get_devices.return_value = [FakeDevice(id="device1"), FakeDevice(id="device2")]

    provider_name = "test_provider"
    backend_name = "non_existent_backend"

    with pytest.raises(QBraidSetupError, match="Device not found"):
        setup_device(provider_name, backend_name)

    assert (
        f"No device matching the name '{backend_name}' found in provider '{provider_name}'."
        in caplog.text
    )
    assert "Devices available: device1, device2" in caplog.text


def test_setup_device_empty_device(mock_provider, patch_load_provider, caplog):
    caplog.set_level(logging.INFO)
    mock_provider.get_device.side_effect = QbraidError()
    mock_provider.get_devices.return_value = [FakeDevice(id="device1"), FakeDevice(id="device2")]

    provider_name = "test_provider"
    backend_name = ""

    with pytest.raises(QBraidSetupError, match="Device not found"):
        setup_device(provider_name, backend_name)

    assert "No device name specified." in caplog.text
    assert "Devices available: device1, device2" in caplog.text


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


def test_estimate_job_quantinuum_defaults(monkeypatch, capsys):
    class DummyParams(BaseModel):
        benchmark_name: str = "WIT"
        num_qubits: int = 6
        shots: int = 16

    captured = {}

    monkeypatch.setattr("os.path.exists", lambda _: True)
    monkeypatch.setattr("metriq_gym.run.load_and_validate", lambda *_: DummyParams())
    monkeypatch.setattr(
        "metriq_gym.run.setup_device",
        lambda *_, **__: SimpleNamespace(id="H1-1", profile=SimpleNamespace(basis_gates=[])),
    )

    # Mock the registry
    mock_registry = MagicMock()
    mock_registry.get_available_benchmarks.return_value = ["WIT"]
    monkeypatch.setattr("metriq_gym.run._lazy_registry", lambda: mock_registry)

    # Mock the benchmark
    mock_benchmark = MagicMock()
    mock_benchmark.estimate_resources_handler.return_value = []
    monkeypatch.setattr("metriq_gym.run.setup_benchmark", lambda *_, **__: mock_benchmark)

    def fake_aggregate(batches, hqc_fn=None):
        counts = GateCounts()
        hqc_value = hqc_fn(counts, 16, 6) if hqc_fn else None
        captured["hqc"] = hqc_value
        circuit_estimate = CircuitEstimate(
            job_index=0,
            circuit_index=0,
            qubit_count=6,
            shots=16,
            gate_counts=counts,
            depth=1,
            hqc=hqc_value,
        )
        return ResourceEstimate(
            job_count=1,
            circuit_count=1,
            total_shots=16,
            max_qubits=6,
            total_gate_counts=counts,
            hqc_total=hqc_value,
            per_circuit=[circuit_estimate],
        )

    monkeypatch.setattr("metriq_gym.run.aggregate_resource_estimates", fake_aggregate)
    monkeypatch.setattr("metriq_gym.run.print_resource_estimate", lambda *_: None)

    args = SimpleNamespace(
        config="foo.json",
        provider="quantinuum",
        device="H1-1",
    )

    estimate_job(args, MagicMock())

    expected = quantinuum_hqc_formula(GateCounts(), 16, 6)
    assert abs(captured["hqc"] - expected) < 1e-6


def test_estimate_job_without_device_wit(monkeypatch, capsys):
    class DummyParams(BaseModel):
        benchmark_name: str = "WIT"
        num_qubits: int = 6
        shots: int = 16

    captured = {}

    monkeypatch.setattr("os.path.exists", lambda *_: True)
    monkeypatch.setattr("metriq_gym.run.load_and_validate", lambda *_: DummyParams())

    def fail_setup(*_args, **_kwargs):
        raise AssertionError("setup_device should not be called when device is omitted")

    monkeypatch.setattr("metriq_gym.run.setup_device", fail_setup)

    # Mock the registry
    mock_registry = MagicMock()
    mock_registry.get_available_benchmarks.return_value = ["WIT"]
    monkeypatch.setattr("metriq_gym.run._lazy_registry", lambda: mock_registry)

    # Mock the benchmark
    def mock_estimate_handler(device):
        captured["device"] = device
        return []

    mock_benchmark = MagicMock()
    mock_benchmark.estimate_resources_handler.side_effect = mock_estimate_handler
    monkeypatch.setattr("metriq_gym.run.setup_benchmark", lambda *_, **__: mock_benchmark)

    def fake_aggregate(batches, hqc_fn=None):
        counts = GateCounts()
        circuit_estimate = CircuitEstimate(
            job_index=0,
            circuit_index=0,
            qubit_count=6,
            shots=16,
            gate_counts=counts,
            depth=1,
            hqc=None,
        )
        return ResourceEstimate(
            job_count=1,
            circuit_count=1,
            total_shots=16,
            max_qubits=6,
            total_gate_counts=counts,
            hqc_total=None,
            per_circuit=[circuit_estimate],
        )

    monkeypatch.setattr("metriq_gym.run.aggregate_resource_estimates", fake_aggregate)

    args = SimpleNamespace(
        config="foo.json",
        provider="quantinuum",
        device=None,
    )

    estimate_job(args, MagicMock())

    output = capsys.readouterr().out
    assert "Resource estimate for WIT" in output
    assert "(no device)" in output
    assert captured["device"] is None


def test_estimate_job_requires_device(monkeypatch, capsys):
    class DummyParams(BaseModel):
        benchmark_name: str = "BSEQ"
        shots: int = 10

    monkeypatch.setattr("os.path.exists", lambda *_: True)
    monkeypatch.setattr("metriq_gym.run.load_and_validate", lambda *_: DummyParams())

    # Mock the registry
    mock_registry = MagicMock()
    mock_registry.get_available_benchmarks.return_value = ["BSEQ"]
    monkeypatch.setattr("metriq_gym.run._lazy_registry", lambda: mock_registry)

    # Mock the benchmark to raise an error when device is None
    mock_benchmark = MagicMock()
    mock_benchmark.estimate_resources_handler.side_effect = ValueError(
        "BSEQ benchmark requires a device to estimate resources."
    )
    monkeypatch.setattr("metriq_gym.run.setup_benchmark", lambda *_, **__: mock_benchmark)

    args = SimpleNamespace(
        config="foo.json",
        provider="aws",
        device=None,
    )

    estimate_job(args, MagicMock())

    output = capsys.readouterr().out
    assert "✗ BSEQ" in output
    assert "requires a device" in output


class DummyResult(BenchmarkResult):
    value: int

    def compute_score(self):
        return BenchmarkScore(value=float(self.value), uncertainty=None)


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
        dispatch_time=datetime.now(),
        result_data={"value": val},
    )


def test_fetch_result_uses_cache_when_no_flag(monkeypatch):
    EXPECTED_CACHED_VALUE = 7
    job = _make_cached_job(EXPECTED_CACHED_VALUE)
    jm = JobManager()
    jm.jobs.append(job)
    args = MagicMock()
    args.no_cache = False
    args.include_raw = False

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

    fetch_output = fetch_result(job, args, jm)
    assert fetch_output.result.value == EXPECTED_CACHED_VALUE
    assert fetch_output.from_cache is True


def test_fetch_result_bypasses_cache_with_flag(monkeypatch):
    EXPECTED_FRESH_VALUE = 42
    CACHED_VALUE = 7
    job = _make_cached_job(CACHED_VALUE)
    jm = JobManager()
    jm.jobs.append(job)
    args = MagicMock()
    args.no_cache = True
    args.include_raw = False

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

    fetch_output = fetch_result(job, args, jm)
    assert fetch_output.result.value == EXPECTED_FRESH_VALUE, (
        "Should fetch fresh value when --no-cache specified"
    )
    assert fetch_output.from_cache is False
    assert job.result_data == {"value": EXPECTED_FRESH_VALUE}, (
        "Cached result_data should be updated"
    )


def test_fetch_result_includes_raw_counts_when_flag_set(monkeypatch):
    """Test that raw counts are returned when include_raw=True."""
    from qbraid.runtime.result_data import GateModelResultData, MeasCount

    EXPECTED_VALUE = 42
    job = _make_cached_job(None)  # No cache, force fresh fetch
    job.result_data = None
    jm = JobManager()
    jm.jobs.append(job)
    args = MagicMock()
    args.no_cache = False
    args.include_raw = True

    # Create a fake quantum job that returns real GateModelResultData
    class FakeQuantumJobWithRealData:
        def __init__(self, job_id, value):
            self.id = job_id
            self._value = value

        def status(self):
            from qbraid.runtime import JobStatus

            return JobStatus.COMPLETED

        def result(self):
            from types import SimpleNamespace

            # Return real GateModelResultData with measurement counts
            data = GateModelResultData(measurement_counts=MeasCount({"00": 512, "11": 488}))
            return SimpleNamespace(data=data)

    # Custom benchmark that handles GateModelResultData
    class DummyBenchmarkWithRealData:
        def poll_handler(self, job_data, result_data, quantum_jobs):
            return DummyResult(value=EXPECTED_VALUE)

    import metriq_gym.run as run_mod

    monkeypatch.setattr(run_mod, "setup_benchmark_result_class", lambda *_: DummyResult)
    monkeypatch.setattr(run_mod, "setup_job_data_class", lambda *_: DummyJobData)
    monkeypatch.setattr(run_mod, "setup_benchmark", lambda *_, **__: DummyBenchmarkWithRealData())
    monkeypatch.setattr(
        run_mod,
        "load_job",
        lambda *_, **__: FakeQuantumJobWithRealData("provider-job-1", EXPECTED_VALUE),
    )
    monkeypatch.setattr(run_mod, "validate_and_create_model", lambda params: params)

    fetch_output = fetch_result(job, args, jm)
    assert fetch_output.raw_results is not None
    assert len(fetch_output.raw_results) == 1
    # Check that the first result is a GateModelResultData and has the expected measurement_counts
    result = fetch_output.raw_results[0]
    assert hasattr(result, "measurement_counts")
    assert dict(result.measurement_counts) == {"00": 512, "11": 488}
    assert fetch_output.from_cache is False


def test_export_raw_debug_data_creates_separate_file(tmp_path):
    """Test that _export_raw_debug_data creates a separate debug JSON file."""
    import json
    from metriq_gym.run import _export_raw_debug_data
    from metriq_gym.constants import JobType

    # Create a mock job
    job = MetriqGymJob(
        id="test-job-123",
        job_type=JobType.WIT,
        params={"benchmark_name": "WIT", "num_qubits": 4},
        data={"provider_job_ids": ["prov-123"], "qubit_chain": [0, 1, 2, 3]},
        provider_name="local",
        device_name="aer_simulator",
        dispatch_time=datetime.now(),
    )

    from qbraid.runtime.result_data import GateModelResultData, MeasCount

    raw_results = [GateModelResultData(measurement_counts=MeasCount({"00": 512, "11": 488}))]
    base_file = tmp_path / "result.json"

    _export_raw_debug_data(str(base_file), job, raw_results)

    debug_file = tmp_path / "result_debug.json"
    assert debug_file.exists()

    with open(debug_file) as f:
        data = json.load(f)

    assert data["job_id"] == "test-job-123"
    assert data["job_type"] == "WIT"
    assert data["params"] == {"benchmark_name": "WIT", "num_qubits": 4}
    assert data["job_data"]["qubit_chain"] == [0, 1, 2, 3]
    # The raw_results are serialized with to_dict(), so check the dict structure
    assert "raw_results" in data
    assert isinstance(data["raw_results"], list)
    assert data["raw_results"][0]["measurement_counts"] == {"00": 512, "11": 488}


def test_replay_from_debug_file_success(tmp_path):
    """Test successful replay from a debug file."""
    import json
    from metriq_gym.run import replay_from_debug_file

    # Create a debug file for QML Kernel benchmark
    debug_data = {
        "job_id": "test-replay-job",
        "job_type": "QML Kernel",
        "params": {
            "benchmark_name": "QML Kernel",
            "num_qubits": 4,
            "shots": 10,
        },
        "job_data": {
            "provider_job_ids": ["prov-123"],
        },
        "raw_results": [
            {
                "measurement_counts": {"0000": 10},
                "shots": 10,
                "num_measured_qubits": 4,
            }
        ],
    }

    debug_file = tmp_path / "debug.json"
    with open(debug_file, "w") as f:
        json.dump(debug_data, f)

    result = replay_from_debug_file(str(debug_file))

    assert result is not None
    # QML Kernel should return accuracy_score
    assert hasattr(result, "accuracy_score")


def test_replay_from_debug_file_missing_file():
    """Test replay with non-existent file."""
    from metriq_gym.run import replay_from_debug_file

    result = replay_from_debug_file("/nonexistent/path/debug.json")
    assert result is None


def test_replay_from_debug_file_invalid_json(tmp_path):
    """Test replay with invalid JSON file."""
    from metriq_gym.run import replay_from_debug_file

    debug_file = tmp_path / "invalid.json"
    with open(debug_file, "w") as f:
        f.write("not valid json {{{")

    result = replay_from_debug_file(str(debug_file))
    assert result is None


def test_replay_from_debug_file_missing_fields(tmp_path):
    """Test replay with missing required fields."""
    import json
    from metriq_gym.run import replay_from_debug_file

    # Missing raw_results field
    debug_data = {
        "job_id": "test-job",
        "job_type": "QML Kernel",
        "params": {"benchmark_name": "QML Kernel"},
        "job_data": {},
    }

    debug_file = tmp_path / "incomplete.json"
    with open(debug_file, "w") as f:
        json.dump(debug_data, f)

    result = replay_from_debug_file(str(debug_file))
    assert result is None


def test_replay_from_debug_file_unknown_job_type(tmp_path):
    """Test replay with unknown job type."""
    import json
    from metriq_gym.run import replay_from_debug_file

    debug_data = {
        "job_id": "test-job",
        "job_type": "Unknown Benchmark Type",
        "params": {},
        "job_data": {},
        "raw_results": [],
    }

    debug_file = tmp_path / "unknown_type.json"
    with open(debug_file, "w") as f:
        json.dump(debug_data, f)

    result = replay_from_debug_file(str(debug_file))
    assert result is None
