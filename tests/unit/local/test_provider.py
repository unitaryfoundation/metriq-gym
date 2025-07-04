import pytest
from metriq_gym.local.provider import LocalProvider


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
