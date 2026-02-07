"""Unit tests for IBMSamplerDevice.

These tests verify the submit() override without requiring IBM credentials
by mocking the underlying Session, Sampler, and backend objects.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.options import TwirlingOptions

from metriq_gym.ibm_sampler.device import IBMSamplerDevice
from qbraid.runtime.ibm.device import QiskitBackend


@pytest.fixture
def mock_device():
    """Create an IBMSamplerDevice with a mocked backend."""
    profile = MagicMock()
    profile.device_id = "ibm_test_backend"
    profile.simulator = False
    profile.instance = None

    mock_service = MagicMock()
    mock_backend = MagicMock()
    mock_backend.name = "ibm_test_backend"
    mock_service.backend.return_value = mock_backend

    with patch.object(QiskitBackend, "__init__", lambda self, **kwargs: None):
        device = IBMSamplerDevice.__new__(IBMSamplerDevice)
        device._backend = mock_backend
        device._service = mock_service
        device._profile = profile
        device._options = MagicMock()

    return device


def _make_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def test_is_subclass_of_qiskit_backend():
    assert issubclass(IBMSamplerDevice, QiskitBackend)


def test_submit_uses_session(mock_device):
    """Verify submit() creates a Session and Sampler."""
    qc = _make_circuit()
    mock_job = MagicMock()
    mock_job.job_id.return_value = "test_job_123"

    with (
        patch("metriq_gym.ibm_sampler.device.Session") as MockSession,
        patch("metriq_gym.ibm_sampler.device.Sampler") as MockSampler,
    ):
        mock_session_instance = MagicMock()
        MockSession.return_value.__enter__ = MagicMock(return_value=mock_session_instance)
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        mock_sampler_instance = MagicMock()
        mock_sampler_instance.run.return_value = mock_job
        MockSampler.return_value = mock_sampler_instance

        result = mock_device.submit([qc], shots=100)

        # Session was created with backend
        MockSession.assert_called_once_with(backend=mock_device._backend)

        # Sampler was created with session and options
        MockSampler.assert_called_once()
        call_kwargs = MockSampler.call_args
        assert call_kwargs.kwargs["mode"] == mock_session_instance
        options = call_kwargs.kwargs["options"]
        assert options.experimental["execution"]["fast_parametric_update"] is True

        # sampler.run was called
        mock_sampler_instance.run.assert_called_once()

        assert result.id == "test_job_123"


def test_submit_with_twirling_options(mock_device):
    """Verify twirling options are forwarded to the SamplerOptions."""
    qc = _make_circuit()
    mock_job = MagicMock()
    mock_job.job_id.return_value = "twirl_job_456"

    twirling = TwirlingOptions(
        num_randomizations=100,
        shots_per_randomization=50,
        enable_gates=True,
    )

    with (
        patch("metriq_gym.ibm_sampler.device.Session") as MockSession,
        patch("metriq_gym.ibm_sampler.device.Sampler") as MockSampler,
    ):
        mock_session_instance = MagicMock()
        MockSession.return_value.__enter__ = MagicMock(return_value=mock_session_instance)
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        mock_sampler_instance = MagicMock()
        mock_sampler_instance.run.return_value = mock_job
        MockSampler.return_value = mock_sampler_instance

        result = mock_device.submit([qc], shots=100, twirling_options=twirling)

        # Verify twirling options were set
        call_kwargs = MockSampler.call_args
        options = call_kwargs.kwargs["options"]
        assert options.twirling == twirling

        assert result.id == "twirl_job_456"


def test_submit_with_parameter_pubs(mock_device):
    """Verify PUBs with parameter bindings are passed through."""
    qc = _make_circuit()
    mock_job = MagicMock()
    mock_job.job_id.return_value = "param_job_789"

    param_values = np.array([[0.1, 0.2], [0.3, 0.4]])
    pubs = [(qc, param_values, 100)]

    with (
        patch("metriq_gym.ibm_sampler.device.Session") as MockSession,
        patch("metriq_gym.ibm_sampler.device.Sampler") as MockSampler,
    ):
        mock_session_instance = MagicMock()
        MockSession.return_value.__enter__ = MagicMock(return_value=mock_session_instance)
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        mock_sampler_instance = MagicMock()
        mock_sampler_instance.run.return_value = mock_job
        MockSampler.return_value = mock_sampler_instance

        result = mock_device.submit(pubs, shots=100)

        # Verify PUBs were passed through to sampler.run
        run_call = mock_sampler_instance.run.call_args
        submitted_pubs = run_call.args[0]
        assert len(submitted_pubs) == 1
        assert submitted_pubs[0][0] is qc
        assert np.array_equal(submitted_pubs[0][1], param_values)
        assert submitted_pubs[0][2] == 100

        assert result.id == "param_job_789"


def test_submit_multiple_pubs(mock_device):
    """Verify multiple PUBs are all submitted."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.measure_all()
    qc2 = QuantumCircuit(2)
    qc2.x(0)
    qc2.measure_all()

    mock_job = MagicMock()
    mock_job.job_id.return_value = "multi_job_000"

    with (
        patch("metriq_gym.ibm_sampler.device.Session") as MockSession,
        patch("metriq_gym.ibm_sampler.device.Sampler") as MockSampler,
    ):
        mock_session_instance = MagicMock()
        MockSession.return_value.__enter__ = MagicMock(return_value=mock_session_instance)
        MockSession.return_value.__exit__ = MagicMock(return_value=False)

        mock_sampler_instance = MagicMock()
        mock_sampler_instance.run.return_value = mock_job
        MockSampler.return_value = mock_sampler_instance

        result = mock_device.submit([qc1, qc2], shots=50)

        run_call = mock_sampler_instance.run.call_args
        pubs = run_call.args[0]
        assert len(pubs) == 2
        assert result.id == "multi_job_000"


def test_submit_without_session(mock_device):
    """Verify use_session=False skips Session and uses backend directly."""
    qc = _make_circuit()
    mock_job = MagicMock()
    mock_job.job_id.return_value = "no_session_job"

    with (
        patch("metriq_gym.ibm_sampler.device.Session") as MockSession,
        patch("metriq_gym.ibm_sampler.device.Sampler") as MockSampler,
    ):
        mock_sampler_instance = MagicMock()
        mock_sampler_instance.run.return_value = mock_job
        MockSampler.return_value = mock_sampler_instance

        result = mock_device.submit([qc], shots=100, use_session=False)

        # Session should NOT have been created
        MockSession.assert_not_called()

        # Sampler should use the backend directly as mode
        call_kwargs = MockSampler.call_args
        assert call_kwargs.kwargs["mode"] == mock_device._backend

        mock_sampler_instance.run.assert_called_once()
        assert result.id == "no_session_job"
