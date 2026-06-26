import pytest
from unittest.mock import MagicMock, patch

from metriq_gym.local.provider import LocalProvider, LocalAerDevice, _local_fake_backend_alias


def fake_backend(name: str):
    backend = MagicMock()
    backend.name = name
    return backend


def test_get_devices_returns_list_with_device():
    """Test that get_devices returns at least the default aer_simulator when no IBM account is configured."""
    provider = LocalProvider()
    with (
        patch.object(provider, "_get_fake_local_backends", return_value={}),
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
    ):
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
        patch.object(provider, "_get_fake_local_backends", return_value={}),
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


def test_get_devices_includes_local_fake_backends_without_ibm_auth():
    """Test that local fake backends are listed without configuring IBM credentials."""
    provider = LocalProvider()
    backend = fake_backend("fake_torino")
    aer_backend = MagicMock()

    with (
        patch.object(provider, "_get_fake_local_backends", return_value={"fake_torino": backend}),
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_service.side_effect = Exception("Not configured")
        mock_aer.from_backend.return_value = aer_backend

        devices = provider.get_devices()

    assert {device.id for device in devices} == {"aer_simulator", "fake_torino"}
    mock_service.assert_called_once()
    mock_aer.from_backend.assert_called_once_with(backend)


def test_local_fake_backend_alias_accepts_ibm_device_names():
    backends = {
        "fake_torino": fake_backend("fake_torino"),
        "fake_sherbrooke": fake_backend("fake_sherbrooke"),
    }

    assert _local_fake_backend_alias("ibm_torino", backends) == "fake_torino"
    assert _local_fake_backend_alias("ibm_sherbrooke", backends) == "fake_sherbrooke"
    assert _local_fake_backend_alias("fake_torino", backends) == "fake_torino"
    assert _local_fake_backend_alias("ibm_unknown", backends) is None
    assert _local_fake_backend_alias("IBM_Torino", backends) is None
    assert _local_fake_backend_alias("ibm-torino", backends) is None
    assert _local_fake_backend_alias("torino", backends) is None


def test_fake_local_backends_are_cached_on_provider():
    provider = LocalProvider()
    backend = fake_backend("fake_torino")

    with patch("metriq_gym.local.provider._local_fake_backends", return_value={"fake_torino": backend}) as fake_backends:
        assert provider._get_fake_local_backends() == {"fake_torino": backend}
        assert provider._get_fake_local_backends() == {"fake_torino": backend}

    fake_backends.assert_called_once_with()


def test_get_device_with_valid_id():
    provider = LocalProvider()
    device = provider.get_device("aer_simulator")
    assert device is provider.device


def test_get_device_with_invalid_id_raises():
    provider = LocalProvider()
    with (
        patch.object(provider, "_get_fake_local_backends", return_value={}),
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        pytest.raises(ValueError, match="Unknown device identifier"),
    ):
        mock_service.side_effect = Exception("Not configured")
        provider.get_device("invalid_id")


def test_get_device_with_noise_model():
    provider = LocalProvider()
    with (
        patch.object(provider, "_get_fake_local_backends", return_value={}),
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


def test_get_device_uses_local_fake_backend_without_ibm_auth():
    provider = LocalProvider()
    backend = fake_backend("fake_torino")
    aer_backend = MagicMock()

    with (
        patch.object(provider, "_get_fake_local_backends", return_value={"fake_torino": backend}),
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_service.side_effect = Exception("Not configured")
        mock_aer.from_backend.return_value = aer_backend

        device = provider.get_device("fake_torino")

    assert isinstance(device, LocalAerDevice)
    mock_service.assert_not_called()
    mock_aer.from_backend.assert_called_once_with(backend)


def test_get_device_uses_local_fake_backend_for_ibm_name_without_ibm_auth():
    provider = LocalProvider()
    backend = fake_backend("fake_torino")
    aer_backend = MagicMock()

    with (
        patch.object(provider, "_get_fake_local_backends", return_value={"fake_torino": backend}),
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_service.side_effect = Exception("Not configured")
        mock_aer.from_backend.return_value = aer_backend

        device = provider.get_device("ibm_torino")

    assert isinstance(device, LocalAerDevice)
    assert device.id == "ibm_torino"
    mock_service.assert_not_called()
    mock_aer.from_backend.assert_called_once_with(backend)


def test_get_device_resolves_real_qiskit_fake_backend_alias_without_ibm_auth(monkeypatch):
    fake_backend_names = sorted(
        backend.name
        for backend in LocalProvider()._get_fake_local_backends().values()
        if isinstance(backend.name, str) and backend.name.startswith("fake_")
    )
    if not fake_backend_names:
        pytest.fail("FakeProviderForBackendV2 returned no fake_*-prefixed backends")

    fake_backend_name = (
        "fake_torino" if "fake_torino" in fake_backend_names else fake_backend_names[0]
    )
    ibm_device_id = fake_backend_name.replace("fake_", "ibm_", 1)

    for key in (
        "QISKIT_IBM_TOKEN",
        "QISKIT_IBM_CHANNEL",
        "QISKIT_IBM_INSTANCE",
        "IBM_QUANTUM_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    class RuntimeServiceShouldNotBeNeeded:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "QiskitRuntimeService should not be required for local fake backend aliases"
            )

    monkeypatch.setattr(
        "metriq_gym.local.provider.QiskitRuntimeService",
        RuntimeServiceShouldNotBeNeeded,
    )

    device = LocalProvider().get_device(ibm_device_id)

    assert isinstance(device, LocalAerDevice)
    assert device.id == ibm_device_id
    assert device.profile.simulator is True


def test_get_device_reuses_cached_local_fake_backend_aliases():
    provider = LocalProvider()
    backend = fake_backend("fake_torino")
    aer_backend = MagicMock()

    with (
        patch.object(provider, "_get_fake_local_backends", return_value={"fake_torino": backend}),
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_service.side_effect = Exception("Not configured")
        mock_aer.from_backend.return_value = aer_backend

        ibm_device = provider.get_device("ibm_torino")
        fake_device = provider.get_device("fake_torino")

    assert ibm_device is fake_device
    assert ibm_device.id == "ibm_torino"
    assert list(provider._devices) == ["fake_torino"]
    mock_service.assert_not_called()
    mock_aer.from_backend.assert_called_once_with(backend)
