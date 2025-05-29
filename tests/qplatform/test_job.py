from unittest.mock import MagicMock
import pytest
from qbraid.runtime import QuantumJob, QiskitJob
from metriq_gym.qplatform.job import execution_time, job_status, JobStatusInfo
from datetime import datetime, timedelta


def test_execution_time_qiskit():
    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job._job = MagicMock()
    execution_spans = MagicMock()
    execution_spans.start = datetime.now()
    execution_spans.stop = execution_spans.start + timedelta(seconds=10)
    qiskit_job._job.result().metadata = {"execution": {"execution_spans": execution_spans}}

    assert execution_time(qiskit_job) == 10.0


def test_execution_time_unsupported():
    mock_job = MagicMock(spec=QuantumJob)
    with pytest.raises(NotImplementedError):
        execution_time(mock_job)


def test_job_status_with_queue_position():
    """Verify status and queue position are extracted correctly from QiskitJob."""
    status_obj = MagicMock()
    status_obj.name = "QUEUED"

    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job.status.return_value = status_obj
    qiskit_job.queue_position.return_value = 3

    info = job_status(qiskit_job)

    assert isinstance(info, JobStatusInfo)
    assert info.status == "QUEUED"
    assert info.queue_position == 3


def test_job_status_without_queue_position():
    """Verify fallback when queue position is unavailable."""
    status_obj = MagicMock()
    status_obj.name = "RUNNING"

    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job.status.return_value = status_obj
    del qiskit_job.queue_position  # simulate method absence

    info = job_status(qiskit_job)

    assert isinstance(info, JobStatusInfo)
    assert info.status == "RUNNING"
    assert info.queue_position is None
