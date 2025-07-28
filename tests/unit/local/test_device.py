import pytest
from unittest.mock import MagicMock, patch

from qiskit import QuantumCircuit
from metriq_gym.local.device import LocalAerDevice
from metriq_gym.local.job import LocalAerJob
from qbraid.runtime import DeviceStatus

N_SHOTS = 1024


@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.run.return_value.result.return_value.get_counts.return_value = {"00": N_SHOTS}
    return backend


def test_status_returns_online(mock_provider, mock_backend):
    with patch("metriq_gym.local.device._make_profile") as mock_make_profile:
        mock_make_profile.return_value = MagicMock(extra={"backend": mock_backend})
        device = LocalAerDevice(provider=mock_provider, backend=mock_backend, device_id="test")
        mock_make_profile.assert_called_once_with(device_id="test", backend=mock_backend)
        assert device.status() == DeviceStatus.ONLINE


def test_submit_returns_local_aer_job(mock_provider, mock_backend):
    with patch("metriq_gym.local.device._make_profile") as mock_make_profile:
        mock_make_profile.return_value = MagicMock(
            device_id="test_device", extra={"backend": mock_backend}
        )
        device = LocalAerDevice(
            provider=mock_provider, device_id="test_device", backend=mock_backend
        )
        input_circuit = QuantumCircuit(2)
        job = device.submit(input_circuit, shots=N_SHOTS)
        assert isinstance(job, LocalAerJob)
        assert job.device == device
        assert job.result().data.measurement_counts == {"00": N_SHOTS}
