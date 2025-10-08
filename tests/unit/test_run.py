import logging
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qbraid import QbraidError
from pydantic import BaseModel

from metriq_gym.resource_estimation import (
    CircuitEstimate,
    GateCounts,
    ResourceEstimate,
    quantinuum_hqc_formula,
)
from metriq_gym.run import dispatch_job, estimate_job, setup_device
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
    monkeypatch.setattr(
        "metriq_gym.run.SUPPORTED_PROVIDERS",
        {"aws", "ibm", "quantinuum", "local"},
    )


@pytest.fixture
def mock_args():
    """Create mock args for testing dispatch functions."""
    args = MagicMock()
    args.provider = "aws"
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

    provider_name = "aws"
    backend_name = "test_backend"

    device = setup_device(provider_name, backend_name)

    mock_provider.get_device.assert_called_once_with(backend_name)
    assert device == mock_device


@patch("metriq_gym.run.load_provider")
def test_setup_device_ibmq_backend_mapping(load_provider_patch, mock_provider, mock_device):
    load_provider_patch.return_value = mock_provider
    mock_provider.get_device.return_value = mock_device

    device = setup_device("ibm", "backend")

    load_provider_patch.assert_called_once_with("ibm")
    assert device == mock_device


def test_setup_device_invalid_provider(caplog):
    caplog.set_level(logging.INFO)

    provider_name = "unsupported_provider"
    backend_name = "whatever_backend"

    with pytest.raises(QBraidSetupError, match="Provider not found"):
        setup_device(provider_name, backend_name)

    assert "Unsupported provider" in caplog.text
    assert "ibm" in caplog.text


def test_setup_device_invalid_device(mock_provider, patch_load_provider, caplog):
    caplog.set_level(logging.INFO)
    mock_provider.get_device.side_effect = QbraidError()
    mock_provider.get_devices.return_value = [FakeDevice(id="device1"), FakeDevice(id="device2")]

    provider_name = "aws"
    backend_name = "non_existent_backend"

    with pytest.raises(QBraidSetupError, match="Device not found"):
        setup_device(provider_name, backend_name)

    # Verify the printed output
    assert (
        f"No device matching the name '{backend_name}' found in provider '{provider_name}'."
        in caplog.text
    )
    assert "Devices available: ['device1', 'device2']" in caplog.text


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

    def fake_estimate(job_type, params, device, hqc_fn=None):
        counts = GateCounts()
        hqc_value = hqc_fn(counts, 16) if hqc_fn else None
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

    monkeypatch.setattr("metriq_gym.run.estimate_resources", fake_estimate)

    args = SimpleNamespace(
        config="foo.json",
        provider="quantinuum",
        device="H1-1",
    )

    estimate_job(args, MagicMock())

    expected = quantinuum_hqc_formula(GateCounts(), 16)
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

    def fake_estimate(job_type, params, device, hqc_fn=None):
        counts = GateCounts()
        captured["device"] = device
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

    monkeypatch.setattr("metriq_gym.run.estimate_resources", fake_estimate)

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

    def fake_estimate(*_args, **_kwargs):
        raise ValueError("BSEQ benchmark requires a device to estimate resources.")

    monkeypatch.setattr("metriq_gym.run.estimate_resources", fake_estimate)

    args = SimpleNamespace(
        config="foo.json",
        provider="aws",
        device=None,
    )

    estimate_job(args, MagicMock())

    output = capsys.readouterr().out
    assert "✗ BSEQ" in output
    assert "requires a device" in output


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
