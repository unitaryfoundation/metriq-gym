import pytest
from unittest.mock import patch, MagicMock
from metriq_gym.local.provider import LocalProvider
from metriq_gym.local.device import LocalAerDevice


def test_get_devices_returns_list_with_device():
    provider = LocalProvider()
    devices = provider.get_devices()
    assert isinstance(devices, list)
    assert len(devices) == 1
    assert devices[0] is provider.device


def test_get_device_with_valid_id():
    provider = LocalProvider()
    device = provider.get_device("aer_simulator")
    assert device is provider.device


def test_get_device_with_invalid_id_raises():
    provider = LocalProvider()
    with pytest.raises(ValueError, match="Unknown device identifier"):
        provider.get_device("invalid_id")


def test_get_device_noisy_raises_without_backend_name():
    """Verifies it raises an error if noise_backend is missing."""
    provider = LocalProvider()
    with pytest.raises(ValueError, match="argument is required"):
        provider.get_device("aer_simulator_noisy")


@patch("qiskit_ibm_runtime.QiskitRuntimeService")
@patch("qiskit_aer.noise.NoiseModel")
@patch("qiskit_aer.AerSimulator")
def test_get_device_noisy_happy_path(mock_aer_simulator, mock_noise_model, mock_runtime_service):
    """Tests that the noisy simulator is constructed and returned correctly."""
    # 1. Setup Mocks
    mock_real_backend = MagicMock()
    mock_real_backend.configuration.return_value.coupling_map = [[0, 1]]

    # Mock the service to return our mock backend
    mock_runtime_service.return_value.backend.return_value = mock_real_backend

    # Mock the NoiseModel class
    mock_noise_model.from_backend.return_value.basis_gates = ["cx", "rz"]

    # 2. Run Code
    provider = LocalProvider()
    device = provider.get_device("aer_simulator_noisy", noise_backend="ibm_fake_brisbane")

    # 3. Assertions
    assert isinstance(device, LocalAerDevice)

    mock_runtime_service.return_value.backend.assert_called_with("ibm_fake_brisbane")

    mock_aer_simulator.assert_called_with(
        noise_model=mock_noise_model.from_backend.return_value,
        coupling_map=[[0, 1]],
        basis_gates=["cx", "rz"],
    )
