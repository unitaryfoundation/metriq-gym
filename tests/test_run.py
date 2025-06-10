
import logging
import pytest
from unittest.mock import MagicMock, patch

from qbraid import QbraidError
from metriq_gym.run import (
    setup_device,
    dispatch_all_benchmarks,
    dispatch_job,
)
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.benchmarks import JobType, get_available_benchmarks
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
    args.all_benchmarks = False
    args.input_file = "test.json"
    args.exclude_benchmarks = None
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
def test_dispatch_all_benchmarks_success(
    mock_exists,
    mock_setup_benchmark,
    mock_load_validate,
    mock_setup_device,
    mock_args,
    mock_job_manager,
    capsys,
):
    """Test successful dispatch of all benchmarks to a single device."""
    # Setup mocks
    mock_args.all_benchmarks = True
    mock_args.exclude_benchmarks = None
    mock_exists.return_value = True

    mock_device = MagicMock()
    mock_setup_device.return_value = mock_device

    mock_params = MagicMock()
    mock_params.model_dump.return_value = {"test": "params"}
    mock_load_validate.return_value = mock_params

    mock_handler = MagicMock()
    # Create real dataclass instances instead of mocks
    mock_job_data = BSEQData(provider_job_ids=["job1", "job2"], shots=1000, num_qubits=5)
    mock_handler.dispatch_handler.return_value = mock_job_data
    mock_setup_benchmark.return_value = mock_handler

    # Execute function
    dispatch_all_benchmarks(mock_args, mock_job_manager)

    # Verify output - DYNAMIC: gets current number of benchmarks
    captured = capsys.readouterr()
    assert "Starting bulk benchmark dispatch..." in captured.out
    
    # Dynamic assertion based on actual number of available benchmarks
    total_benchmarks = len(get_available_benchmarks())
    assert f"Successfully dispatched {total_benchmarks}/{total_benchmarks} benchmarks." in captured.out

    # Verify all benchmarks were processed
    assert mock_job_manager.add_job.call_count == total_benchmarks


@patch("metriq_gym.run.setup_device")
@patch("metriq_gym.run.load_and_validate")
@patch("os.path.exists")
def test_dispatch_all_benchmarks_with_exclusions(
    mock_exists, mock_load_validate, mock_setup_device, mock_args, mock_job_manager, capsys
):
    """Test dispatch of all benchmarks excluding specific ones."""
    # Setup mocks
    mock_args.all_benchmarks = True
    mock_args.exclude_benchmarks = ["CLOPS", "QML Kernel"]
    mock_exists.return_value = True

    mock_device = MagicMock()
    mock_setup_device.return_value = mock_device

    mock_params = MagicMock()
    mock_params.model_dump.return_value = {"test": "params"}
    mock_load_validate.return_value = mock_params

    # Mock the benchmark and handler
    with patch("metriq_gym.run.setup_benchmark") as mock_setup_benchmark:
        mock_handler = MagicMock()
        mock_job_data = BSEQData(provider_job_ids=["job1"], shots=1000, num_qubits=5)
        mock_handler.dispatch_handler.return_value = mock_job_data
        mock_setup_benchmark.return_value = mock_handler

        # Execute function
        dispatch_all_benchmarks(mock_args, mock_job_manager)

        # Verify output shows exclusions - DYNAMIC calculation
        captured = capsys.readouterr()
        assert "Excluding benchmarks: ['CLOPS', 'QML Kernel']" in captured.out
        
        # Calculate expected number dynamically
        total_benchmarks = len(get_available_benchmarks())
        excluded_count = len([jt for jt in get_available_benchmarks() 
                             if jt.value in ["CLOPS", "QML Kernel"]])
        expected_count = total_benchmarks - excluded_count
        
        assert f"Running {expected_count} benchmarks" in captured.out


@patch("os.path.exists")
def test_dispatch_all_benchmarks_missing_example_file(
    mock_exists, mock_args, mock_job_manager, capsys
):
    """Test behavior when example file is missing for a benchmark."""
    # Setup mocks
    mock_args.all_benchmarks = True
    mock_args.exclude_benchmarks = None
    mock_exists.return_value = False  # Simulate missing files

    # Mock setup_device to avoid provider validation error
    with patch("metriq_gym.run.setup_device") as mock_setup_device:
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device

        # Execute function
        dispatch_all_benchmarks(mock_args, mock_job_manager)

        # Verify output shows file not found errors - DYNAMIC
        captured = capsys.readouterr()
        assert "Example file not found" in captured.out
        
        # Dynamic assertion
        total_benchmarks = len(get_available_benchmarks())
        assert f"Successfully dispatched 0/{total_benchmarks} benchmarks." in captured.out


@patch("metriq_gym.run.dispatch_all_benchmarks")
def test_dispatch_job_calls_all_benchmarks(mock_dispatch_all, mock_args, mock_job_manager):
    """Test that dispatch_job calls dispatch_all_benchmarks when --all-benchmarks is used."""
    # Setup args for all benchmarks mode
    mock_args.all_benchmarks = True

    # Execute function
    dispatch_job(mock_args, mock_job_manager)

    # Verify dispatch_all_benchmarks was called
    mock_dispatch_all.assert_called_once_with(mock_args, mock_job_manager)


@patch("metriq_gym.run.load_and_validate")
@patch("metriq_gym.run.setup_device")
@patch("metriq_gym.run.setup_benchmark")
def test_dispatch_job_calls_single_benchmark(
    mock_setup_benchmark, mock_setup_device, mock_load_validate, mock_args, mock_job_manager
):
    """Test that dispatch_job calls single benchmark dispatch when input_file is provided."""
    # Setup args for single benchmark mode
    mock_args.all_benchmarks = False
    mock_args.input_file = "test.json"

    # Setup mocks for single benchmark flow
    mock_device = MagicMock()
    mock_setup_device.return_value = mock_device

    mock_params = MagicMock()
    mock_params.benchmark_name = "BSEQ"
    mock_params.model_dump.return_value = {"test": "params"}
    mock_load_validate.return_value = mock_params

    # Setup benchmark mock with real dataclass
    mock_handler = MagicMock()
    mock_job_data = BSEQData(provider_job_ids=["job1"], shots=1000, num_qubits=5)
    mock_handler.dispatch_handler.return_value = mock_job_data
    mock_setup_benchmark.return_value = mock_handler

    # Mock JobType enum
    with patch("metriq_gym.run.JobType") as mock_job_type:
        mock_job_type.return_value = JobType.BSEQ

        # Execute function
        dispatch_job(mock_args, mock_job_manager)

    # Verify single benchmark functions were called
    mock_load_validate.assert_called_once_with("test.json")
    mock_setup_device.assert_called_once()


def test_dispatch_job_missing_input_file(mock_args, mock_job_manager, caplog):
    """Test that dispatch_job logs error when input_file is missing and --all-benchmarks is not used."""
    caplog.set_level(logging.INFO)

    # Setup args with missing input file
    mock_args.all_benchmarks = False
    mock_args.input_file = None

    # Execute function
    dispatch_job(mock_args, mock_job_manager)

    # Verify error is logged
    assert "input_file is required when not using --all-benchmarks" in caplog.text


def test_get_available_benchmarks_dynamic():
    """Test that get_available_benchmarks returns current JobType list."""
    available = get_available_benchmarks()
    
    # Verify it returns a list of JobType
    assert isinstance(available, list)
    assert all(isinstance(jt, JobType) for jt in available)
    
    # Verify it includes known benchmarks
    benchmark_values = [jt.value for jt in available]
    assert "BSEQ" in benchmark_values
    assert "CLOPS" in benchmark_values
    assert "Quantum Volume" in benchmark_values
    assert "QML Kernel" in benchmark_values
    assert "Wormhole" in benchmark_values
    
    # Verify it matches JobType enum exactly
    assert len(available) == len(JobType)
    assert set(available) == set(JobType)