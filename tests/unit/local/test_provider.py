import pytest
from unittest.mock import MagicMock, patch

from metriq_gym.local.provider import LocalProvider, LocalAerDevice


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

        device_name = mock_service.return_value.backends.return_value[0].name
        device = provider.get_device(device_name)

        assert isinstance(device, LocalAerDevice)
        mock_service.return_value.backend.assert_called_once_with(device_name)
        mock_aer.from_backend.assert_called_once_with(backend)
        assert provider.get_device(device_name) is device
