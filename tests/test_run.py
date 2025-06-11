import logging
import pytest
from unittest.mock import MagicMock, patch

from qbraid import QbraidError
from metriq_gym.run import (
    setup_device,
    dispatch_job,
)
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.benchmarks.bseq import BSEQData


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


@patch("metriq_gym.run.setup_device")
@patch("metriq_gym.run.load_and_validate")
@patch("metriq_gym.run.setup_benchmark")
@patch("os.path.exists")
def test_dispatch_multiple_benchmarks_success(
    mock_exists,
    mock_setup_benchmark,
    mock_load_validate,
    mock_setup_device,
    mock_args,
    mock_job_manager,
    capsys,
):
    """Test successful dispatch of multiple benchmark configuration files."""
    # Setup mocks
    mock_args.benchmark_configs = ["bseq.json", "clops.json"]
    mock_exists.return_value = True

    mock_device = MagicMock()
    mock_setup_device.return_value = mock_device

    # Mock different benchmark params for each file
    mock_bseq_params = MagicMock()
    mock_bseq_params.benchmark_name = "BSEQ"
    mock_bseq_params.model_dump.return_value = {"benchmark_name": "BSEQ", "shots": 1000}

    mock_clops_params = MagicMock()
    mock_clops_params.benchmark_name = "CLOPS"
    mock_clops_params.model_dump.return_value = {"benchmark_name": "CLOPS", "width": 4}

    mock_load_validate.side_effect = [mock_bseq_params, mock_clops_params]

    mock_handler = MagicMock()
    mock_job_data = BSEQData(provider_job_ids=["job1"], shots=1000, num_qubits=5)
    mock_handler.dispatch_handler.return_value = mock_job_data
    mock_setup_benchmark.return_value = mock_handler

    # Execute function
    dispatch_job(mock_args, mock_job_manager)

    # Verify output
    captured = capsys.readouterr()
    assert "Starting job dispatch..." in captured.out
    assert "Successfully dispatched 2/2 benchmarks." in captured.out
    assert "BSEQ (bseq.json) dispatched with ID:" in captured.out
    assert "CLOPS (clops.json) dispatched with ID:" in captured.out

    # Verify all benchmarks were processed
    assert mock_job_manager.add_job.call_count == 2


@patch("metriq_gym.run.setup_device")
@patch("metriq_gym.run.load_and_validate")
@patch("metriq_gym.run.setup_benchmark")
@patch("os.path.exists")
def test_dispatch_duplicate_benchmark_types(
    mock_exists,
    mock_setup_benchmark,
    mock_load_validate,
    mock_setup_device,
    mock_args,
    mock_job_manager,
    capsys,
):
    """Test dispatch of same benchmark type with different configurations."""
    # Setup mocks for same benchmark type, different configs
    mock_args.benchmark_configs = ["bseq_config1.json", "bseq_config2.json"]
    mock_exists.return_value = True

    mock_device = MagicMock()
    mock_setup_device.return_value = mock_device

    # Mock different BSEQ configurations
    mock_bseq_params1 = MagicMock()
    mock_bseq_params1.benchmark_name = "BSEQ"
    mock_bseq_params1.model_dump.return_value = {"benchmark_name": "BSEQ", "shots": 1000}

    mock_bseq_params2 = MagicMock()
    mock_bseq_params2.benchmark_name = "BSEQ"
    mock_bseq_params2.model_dump.return_value = {"benchmark_name": "BSEQ", "shots": 5000}

    mock_load_validate.side_effect = [mock_bseq_params1, mock_bseq_params2]

    mock_handler = MagicMock()
    mock_job_data = BSEQData(provider_job_ids=["job1"], shots=1000, num_qubits=5)
    mock_handler.dispatch_handler.return_value = mock_job_data
    mock_setup_benchmark.return_value = mock_handler

    # Execute function
    dispatch_job(mock_args, mock_job_manager)

    # Verify output
    captured = capsys.readouterr()
    assert "Successfully dispatched 2/2 benchmarks." in captured.out
    assert "BSEQ (bseq_config1.json) dispatched with ID:" in captured.out
    assert "BSEQ (bseq_config2.json) dispatched with ID:" in captured.out

    # Verify both jobs were processed
    assert mock_job_manager.add_job.call_count == 2


@patch("os.path.exists")
def test_dispatch_missing_config_file(mock_exists, mock_args, mock_job_manager, capsys):
    """Test behavior when configuration file is missing."""
    # Setup mocks
    mock_args.benchmark_configs = ["missing.json", "also_missing.json"]
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
        assert "Successfully dispatched 0/2 benchmarks." in captured.out


@patch("metriq_gym.run.setup_device")
@patch("metriq_gym.run.load_and_validate")
@patch("os.path.exists")
def test_dispatch_mixed_success_failure(
    mock_exists, mock_load_validate, mock_setup_device, mock_args, mock_job_manager, capsys
):
    """Test dispatch with mix of successful and failed configurations."""
    # Setup mocks
    mock_args.benchmark_configs = ["good.json", "bad.json"]
    mock_exists.return_value = True

    mock_device = MagicMock()
    mock_setup_device.return_value = mock_device

    # First file succeeds, second fails
    mock_good_params = MagicMock()
    mock_good_params.benchmark_name = "BSEQ"
    mock_good_params.model_dump.return_value = {"benchmark_name": "BSEQ"}

    mock_load_validate.side_effect = [mock_good_params, ValueError("Invalid configuration")]

    # Mock successful benchmark setup for first file
    with patch("metriq_gym.run.setup_benchmark") as mock_setup_benchmark:
        mock_handler = MagicMock()
        mock_job_data = BSEQData(provider_job_ids=["job1"], shots=1000, num_qubits=5)
        mock_handler.dispatch_handler.return_value = mock_job_data
        mock_setup_benchmark.return_value = mock_handler

        # Execute function
        dispatch_job(mock_args, mock_job_manager)

        # Verify output shows mixed results
        captured = capsys.readouterr()
        assert "Successfully dispatched 1/2 benchmarks." in captured.out
        assert "BSEQ (good.json) dispatched with ID:" in captured.out
        assert "bad.json failed: ValueError: Invalid configuration" in captured.out
