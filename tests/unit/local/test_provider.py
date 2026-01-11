import pytest
from unittest.mock import MagicMock, patch

from metriq_gym.local.provider import LocalProvider, LocalAerDevice


def test_get_devices_returns_list_with_device():
    """Test that get_devices returns at least the default aer_simulator when no IBM account is configured."""
    provider = LocalProvider()
    with patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service:
        # Simulate QiskitRuntimeService not configured
        mock_service.side_effect = Exception("Not configured")
        devices = provider.get_devices()
        assert isinstance(devices, list)
        assert len(devices) == 1
        assert devices[0] is provider.device


def test_get_devices_includes_ibm_backends_when_available():
    """Test that get_devices includes IBM backends when QiskitRuntimeService is configured."""
    provider = LocalProvider()
    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        # Create mock backends
        mock_backend1 = MagicMock()
        mock_backend1.name = "fake_manila"
        mock_backend2 = MagicMock()
        mock_backend2.name = "fake_jakarta"

        mock_service.return_value.backends.return_value = [mock_backend1, mock_backend2]
        mock_service.return_value.backend.side_effect = lambda name: (
            mock_backend1 if name == "fake_manila" else mock_backend2
        )

        # Mock AerSimulator.from_backend to return a valid backend
        mock_aer.from_backend.return_value = MagicMock()

        devices = provider.get_devices()

        assert isinstance(devices, list)
        # Should have aer_simulator + 2 fake backends
        assert len(devices) == 3
        assert devices[0] is provider.device
        # Verify the other devices are in the cache
        assert "fake_manila" in provider._devices
        assert "fake_jakarta" in provider._devices


def test_get_device_with_valid_id():
    provider = LocalProvider()
    device = provider.get_device("aer_simulator")
    assert device is provider.device


def test_get_device_with_invalid_id_raises():
    provider = LocalProvider()
    with pytest.raises(ValueError, match="Unknown device identifier"):
        provider.get_device("invalid_id")


def test_get_device_with_noise_model():
    provider = LocalProvider()
    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        backend = MagicMock()
        backend.name = "fake_backend"
        mock_service.return_value.backend.return_value = backend
        aer_backend = MagicMock()
        mock_aer.from_backend.return_value = aer_backend

        device_name = "fake_backend"
        device = provider.get_device(device_name)

        assert isinstance(device, LocalAerDevice)
        mock_service.return_value.backend.assert_called_once_with(device_name)
        mock_aer.from_backend.assert_called_once_with(backend)
        assert provider.get_device(device_name) is device
