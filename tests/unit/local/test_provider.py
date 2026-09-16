import logging
from unittest.mock import MagicMock, patch

import pytest
from qbraid import QbraidError
from qiskit_aer import AerSimulator

from metriq_gym.local.provider import (
    LocalProvider,
    LocalAerDevice,
    _aer_custom_instructions,
    UnknownLocalDeviceError,
)
from metriq_gym.qplatform.device import connectivity_graph, version


class _MockCouplingMap:
    def get_edges(self) -> list[tuple[int, int]]:
        return [(0, 1), (1, 2)]


class _MockBackendV2WithoutConfiguration:
    name = "ibm_torino"
    num_qubits = 3
    operation_names = ["ecr", "id", "rz", "sx", "x"]
    coupling_map = _MockCouplingMap()
    max_circuits = 100
    description = None

    def __init__(self, backend_version: str | None = "1.2.3") -> None:
        self.backend_version = backend_version

    def configuration(self):
        raise AssertionError("BackendV2 listing should not require configuration()")


@pytest.fixture(autouse=True)
def clear_aer_helper_caches():
    _aer_custom_instructions.cache_clear()
    yield
    _aer_custom_instructions.cache_clear()


def _mock_backend(name: str, num_qubits: int = 5) -> MagicMock:
    backend = MagicMock()
    backend.name = name
    backend.backend_version = "1.0.0"
    backend.num_qubits = num_qubits
    backend.operation_names = ["id", "x", "sx", "rz", "cx", "measure"]
    backend.coupling_map = None
    backend.max_circuits = None
    backend.description = None
    config = MagicMock()
    config.num_qubits = num_qubits
    config.n_qubits = num_qubits
    config.basis_gates = ["id", "x", "sx", "rz", "cx"]
    config.backend_version = "1.0.0"
    config.coupling_map = None
    backend.configuration.return_value = config
    return backend


def _mock_aer_backend(num_qubits: int = 5) -> MagicMock:
    backend = MagicMock()
    config = MagicMock()
    config.num_qubits = num_qubits
    config.n_qubits = num_qubits
    config.basis_gates = ["id", "x", "sx", "rz", "cx"]
    config.backend_version = "1.0.0"
    config.coupling_map = None
    backend.configuration.return_value = config
    return backend


def test_get_devices_returns_list_with_device():
    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
    ):
        provider = LocalProvider()
        # Simulate QiskitRuntimeService not configured
        mock_service.side_effect = ValueError("Not configured")
        mock_fake_provider.return_value.backends.return_value = []
        devices = provider.get_devices()
        assert isinstance(devices, list)
        assert len(devices) == 1
        assert devices[0] is provider.device


def test_get_devices_includes_fake_backends_without_ibm_account():
    """Test that local fake backends are listed without IBM Runtime authentication."""
    mock_backend = _mock_backend("fake_torino")

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_service.side_effect = ValueError("Not configured")
        mock_fake_provider.return_value.backends.return_value = [mock_backend]
        mock_aer.from_backend.return_value = MagicMock()

        provider = LocalProvider()
        devices = provider.get_devices()

        assert len(devices) == 2
        assert devices[0] is provider.device
        assert "fake_torino" in provider._devices
        assert provider._devices["fake_torino"] in devices
        mock_service.assert_called_once()
        mock_aer.from_backend.assert_not_called()


def test_get_devices_includes_ibm_backends_when_available():
    """Test that get_devices includes IBM backends when QiskitRuntimeService is configured."""
    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        # Create mock backends
        mock_backend1 = _mock_backend("ibm_manila")
        mock_backend2 = _mock_backend("ibm_jakarta")

        mock_fake_provider.return_value.backends.return_value = []
        mock_service.return_value.backends.return_value = [mock_backend1, mock_backend2]

        provider = LocalProvider()
        devices = provider.get_devices()

        assert isinstance(devices, list)
        # Should have aer_simulator + 2 runtime backends
        assert len(devices) == 3
        assert devices[0] is provider.device
        # Verify the other devices are in the cache
        assert "ibm_manila" in provider._devices
        assert "ibm_jakarta" in provider._devices
        mock_service.return_value.backend.assert_not_called()
        mock_aer.from_backend.assert_not_called()


def test_get_devices_deduplicates_runtime_backend_with_fake_twin():
    """Authenticated listings prefer the live IBM backend over its fake twin."""
    fake_backend = _mock_backend("fake_torino", num_qubits=133)
    runtime_backend = _mock_backend("ibm_torino", num_qubits=133)

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_fake_provider.return_value.backends.return_value = [fake_backend]
        mock_service.return_value.backends.return_value = [runtime_backend]

        provider = LocalProvider()
        devices = provider.get_devices()

        assert len(devices) == 2
        assert "ibm_torino" in provider._devices
        assert "fake_torino" not in provider._devices
        assert provider._devices["ibm_torino"] in devices
        mock_aer.from_backend.assert_not_called()


def test_get_devices_handles_runtime_backend_without_configuration():
    """BackendV2 runtime listings use attributes instead of deprecated configuration()."""
    fake_backend = _mock_backend("fake_torino", num_qubits=133)
    runtime_backend = _MockBackendV2WithoutConfiguration()

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch(
            "metriq_gym.local.provider._aer_custom_instructions",
            return_value=["delay", "reset"],
        ),
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_fake_provider.return_value.backends.return_value = [fake_backend]
        mock_service.return_value.backends.return_value = [runtime_backend]

        provider = LocalProvider()
        devices = provider.get_devices()
        device = provider._devices["ibm_torino"]

        assert len(devices) == 2
        assert "fake_torino" not in provider._devices
        assert device in devices
        assert device.profile.basis_gates == {
            "ecr",
            "id",
            "rz",
            "sx",
            "x",
            "delay",
            "reset",
        }
        assert version(device) == "1.2.3"
        assert sorted(connectivity_graph(device).edge_list()) == [(0, 1), (1, 2)]
        mock_aer.from_backend.assert_not_called()


def test_get_devices_handles_runtime_backend_with_none_backend_version():
    """BackendV2 listings with default None versions still avoid configuration()."""
    fake_backend = _mock_backend("fake_torino", num_qubits=133)
    runtime_backend = _MockBackendV2WithoutConfiguration(backend_version=None)

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch(
            "metriq_gym.local.provider._aer_custom_instructions",
            return_value=["delay", "reset"],
        ),
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_fake_provider.return_value.backends.return_value = [fake_backend]
        mock_service.return_value.backends.return_value = [runtime_backend]

        provider = LocalProvider()
        devices = provider.get_devices()
        device = provider._devices["ibm_torino"]

        assert len(devices) == 2
        assert "fake_torino" not in provider._devices
        assert device in devices
        assert device.profile.basis_gates == {
            "ecr",
            "id",
            "rz",
            "sx",
            "x",
            "delay",
            "reset",
        }
        assert version(device) == "0.0.0"
        assert sorted(connectivity_graph(device).edge_list()) == [(0, 1), (1, 2)]
        mock_aer.from_backend.assert_not_called()


def test_get_devices_logs_runtime_service_failures(caplog):
    caplog.set_level(logging.DEBUG, logger="metriq_gym.local.provider")

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
    ):
        mock_service.side_effect = ValueError("Not configured")
        mock_fake_provider.return_value.backends.return_value = []

        provider = LocalProvider()
        provider.get_devices()

    assert "IBM Runtime service is unavailable for local listing." in caplog.text


def test_fake_backend_registry_is_loaded_lazily():
    with patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider:
        provider = LocalProvider()
        provider.get_device("aer_simulator")

    mock_fake_provider.assert_not_called()


def test_fake_local_backends_are_cached_on_provider():
    mock_backend = _mock_backend("fake_torino")

    with patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider:
        mock_fake_provider.return_value.backends.return_value = [mock_backend]

        provider = LocalProvider()
        assert provider._get_fake_local_backends() == {"fake_torino": mock_backend}
        assert provider._get_fake_local_backends() == {"fake_torino": mock_backend}

    mock_fake_provider.assert_called_once()


def test_unknown_local_device_error_preserves_old_value_error_contract():
    error = UnknownLocalDeviceError("Unknown device identifier")

    assert isinstance(error, QbraidError)
    assert isinstance(error, ValueError)


def test_get_device_with_valid_id():
    provider = LocalProvider()
    device = provider.get_device("aer_simulator")
    assert device is provider.device


def test_get_device_with_invalid_id_raises():
    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
    ):
        provider = LocalProvider()
        mock_fake_provider.return_value.backends.return_value = []
        mock_service.return_value.backend.side_effect = ValueError("Unknown backend")

        with pytest.raises(UnknownLocalDeviceError, match="Unknown device identifier"):
            provider.get_device("invalid_id")

        mock_service.return_value.backend.assert_called_once_with("ibm_invalid_id")


def test_get_device_with_noise_model():
    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        backend = _mock_backend("ibm_backend")
        mock_fake_provider.return_value.backends.return_value = []
        mock_service.return_value.backend.return_value = backend
        aer_backend = _mock_aer_backend()
        mock_aer.from_backend.return_value = aer_backend

        provider = LocalProvider()
        device_name = "ibm_backend"
        device = provider.get_device(device_name)

        assert isinstance(device, LocalAerDevice)
        mock_service.return_value.backend.assert_called_once_with(device_name)
        mock_aer.from_backend.assert_called_once_with(backend)
        assert provider.get_device(device_name) is device


def test_get_device_uses_runtime_backend_before_fake_alias_when_ibm_account_available():
    fake_backend = _mock_backend("fake_torino", num_qubits=133)
    runtime_backend = _mock_backend("ibm_torino", num_qubits=133)

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_fake_provider.return_value.backends.return_value = [fake_backend]
        mock_service.return_value.backend.return_value = runtime_backend
        aer_backend = _mock_aer_backend(num_qubits=133)
        mock_aer.from_backend.return_value = aer_backend

        provider = LocalProvider()
        device = provider.get_device("ibm_torino")

        assert isinstance(device, LocalAerDevice)
        mock_service.return_value.backend.assert_called_once_with("ibm_torino")
        mock_aer.from_backend.assert_called_once_with(runtime_backend)
        assert provider.get_device("ibm_torino") is device


def test_get_device_uses_fake_backend_alias_without_ibm_account():
    mock_backend = _mock_backend("fake_torino", num_qubits=133)

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_fake_provider.return_value.backends.return_value = [mock_backend]
        mock_service.return_value.backend.side_effect = ValueError("Not configured")
        aer_backend = _mock_aer_backend(num_qubits=133)
        mock_aer.from_backend.return_value = aer_backend

        provider = LocalProvider()
        device = provider.get_device("ibm_torino")
        cached_device = provider.get_device("ibm_torino")

        assert isinstance(device, LocalAerDevice)
        mock_service.return_value.backend.assert_called_once_with("ibm_torino")
        mock_aer.from_backend.assert_called_once_with(mock_backend)
        assert cached_device is device
        assert list(provider._devices) == ["fake_torino"]


def test_get_device_reuses_cached_local_fake_backend_aliases():
    mock_backend = _mock_backend("fake_torino", num_qubits=133)

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_fake_provider.return_value.backends.return_value = [mock_backend]
        mock_service.return_value.backend.side_effect = ValueError("Not configured")
        mock_aer.from_backend.return_value = _mock_aer_backend(num_qubits=133)

        provider = LocalProvider()
        ibm_device = provider.get_device("ibm_torino")
        fake_device = provider.get_device("fake_torino")
        bare_device = provider.get_device("torino")

    assert ibm_device is fake_device is bare_device
    assert ibm_device.id == "ibm_torino"
    assert list(provider._devices) == ["fake_torino"]
    mock_service.return_value.backend.assert_called_once_with("ibm_torino")
    mock_aer.from_backend.assert_called_once_with(mock_backend)


@pytest.mark.parametrize(
    ("device_id", "expected_device_id", "expects_runtime_lookup"),
    [
        ("IBM-Torino", "ibm_torino", True),
        ("torino", "ibm_torino", True),
        ("fake-torino", "fake_torino", False),
    ],
)
def test_get_device_normalizes_fake_backend_aliases(
    device_id: str, expected_device_id: str, expects_runtime_lookup: bool
):
    mock_backend = _mock_backend("fake_torino", num_qubits=133)

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_fake_provider.return_value.backends.return_value = [mock_backend]
        mock_service.return_value.backend.side_effect = ValueError("Not configured")
        mock_aer.from_backend.return_value = _mock_aer_backend(num_qubits=133)

        provider = LocalProvider()
        device = provider.get_device(device_id)

        assert isinstance(device, LocalAerDevice)
        assert device.id == expected_device_id
        mock_aer.from_backend.assert_called_once_with(mock_backend)
        if expects_runtime_lookup:
            mock_service.return_value.backend.assert_called_once_with("ibm_torino")
        else:
            mock_service.assert_not_called()


def test_get_device_missing_fake_alias_attempts_runtime_then_raises():
    mock_backend = _mock_backend("fake_torino")

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_fake_provider.return_value.backends.return_value = [mock_backend]
        mock_service.return_value.backend.side_effect = ValueError("Unknown backend")

        provider = LocalProvider()
        with pytest.raises(UnknownLocalDeviceError, match="Unknown device identifier"):
            provider.get_device("ibm_doesnotexist")

        mock_service.return_value.backend.assert_called_once_with("ibm_doesnotexist")
        mock_aer.from_backend.assert_not_called()


def test_get_devices_listed_fake_backend_constructs_aer_on_submit():
    mock_backend = _mock_backend("fake_torino", num_qubits=133)
    aer_backend = _mock_aer_backend(num_qubits=133)
    aer_backend.run.return_value.result.return_value.get_counts.return_value = {"0": 1}

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
        patch("metriq_gym.local.device.transpile", side_effect=lambda circuits, **_: circuits),
    ):
        mock_service.side_effect = ValueError("Not configured")
        mock_fake_provider.return_value.backends.return_value = [mock_backend]
        mock_aer.from_backend.return_value = aer_backend

        provider = LocalProvider()
        provider.get_devices()
        device = provider._devices["fake_torino"]

        mock_aer.from_backend.assert_not_called()
        job = device.submit(MagicMock(), shots=1)

        mock_aer.from_backend.assert_called_once_with(mock_backend)
        assert job.result().data.measurement_counts == {"0": 1}


def test_listed_fake_backend_profile_matches_eager_aer_configuration():
    """A listed lazy fake backend should expose the same profile fields as eager Aer."""
    fake_backend = next(
        backend
        for backend in LocalProvider._load_fake_local_backends().values()
        if backend.name == "fake_torino"
    )

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
    ):
        mock_service.side_effect = ValueError("Not configured")
        mock_fake_provider.return_value.backends.return_value = [fake_backend]
        listed_provider = LocalProvider()
        listed_provider.get_devices()
        listed_device = listed_provider.get_device("fake_torino")

    lazy_backend = listed_device._backend
    lazy_config = lazy_backend.configuration()
    eager_config = AerSimulator.from_backend(fake_backend).configuration()

    assert lazy_backend._aer_backend is None
    missing = object()
    for field in (
        "backend_name",
        "backend_version",
        "n_qubits",
        "num_qubits",
        "basis_gates",
        "gates",
        "max_shots",
        "coupling_map",
        "max_experiments",
        "description",
    ):
        assert getattr(lazy_config, field, missing) == getattr(eager_config, field, missing)


def test_listed_fake_backend_forwards_unknown_attributes_to_materialized_aer():
    mock_backend = _mock_backend("fake_torino", num_qubits=133)
    aer_backend = _mock_aer_backend(num_qubits=133)
    aer_backend.status.return_value = "online"

    with (
        patch("metriq_gym.local.provider.QiskitRuntimeService") as mock_service,
        patch("metriq_gym.local.provider.FakeProviderForBackendV2") as mock_fake_provider,
        patch("metriq_gym.local.provider.AerSimulator") as mock_aer,
    ):
        mock_service.side_effect = ValueError("Not configured")
        mock_fake_provider.return_value.backends.return_value = [mock_backend]
        mock_aer.from_backend.return_value = aer_backend

        provider = LocalProvider()
        provider.get_devices()
        device = provider._devices["fake_torino"]

        mock_aer.from_backend.assert_not_called()
        assert device._backend.status() == "online"

    mock_aer.from_backend.assert_called_once_with(mock_backend)


def test_get_device_resolves_real_qiskit_fake_backend_alias_without_ibm_auth(monkeypatch):
    fake_backend_names = sorted(
        backend.name
        for backend in LocalProvider._load_fake_local_backends().values()
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

    class RuntimeServiceWithoutAccount:
        def __init__(self, *args, **kwargs):
            pass

        def backend(self, *_args, **_kwargs):
            raise ValueError("No IBM account configured")

    monkeypatch.setattr(
        "metriq_gym.local.provider.QiskitRuntimeService",
        RuntimeServiceWithoutAccount,
    )

    device = LocalProvider().get_device(ibm_device_id)

    assert isinstance(device, LocalAerDevice)
    assert device.id == ibm_device_id
    assert device.profile.simulator is True
