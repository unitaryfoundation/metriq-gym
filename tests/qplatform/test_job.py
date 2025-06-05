from unittest.mock import MagicMock
import pytest
from qbraid.runtime import QuantumJob, QiskitJob, BraketQuantumTask
from metriq_gym.qplatform.job import execution_time, job_status, JobStatusInfo, JobStatus
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
    """Verify execution_time raises NotImplementedError."""
    quantum_job = MagicMock(spec=QuantumJob)
    with pytest.raises(NotImplementedError):
        execution_time(quantum_job)


def test_job_status_with_queue_position():
    """Verify status and queue position are extracted correctly from QiskitJob."""
    status_obj = MagicMock()
    status_obj.name = "QUEUED"

    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job.status.return_value = status_obj
    qiskit_job.queue_position.return_value = 3

    info = job_status(qiskit_job)

    assert isinstance(info, JobStatusInfo)
    assert info.status == JobStatus.QUEUED
    assert info.queue_position == 3
    assert info.queue_type is None
    assert info.message is None


def test_job_status_with_queue_type():
    """Verify status and queue type are extracted correctly from BraketQuantumTask."""
    status_obj = MagicMock()
    status_obj.name = "QUEUED"

    queue_info = MagicMock()
    queue_info.position = 5
    queue_info.queue_type = "PRIORITY"
    queue_info.message = "High priority task"

    braket_job = MagicMock(spec=BraketQuantumTask)
    braket_job.status.return_value = status_obj
    braket_job.queue_position.return_value = queue_info

    info = job_status(braket_job)

    assert isinstance(info, JobStatusInfo)
    assert info.status == JobStatus.QUEUED
    assert info.queue_position == 5
    assert info.queue_type == "PRIORITY"
    assert info.message == "High priority task"


def test_job_status_without_queue_position():
    """Verify fallback when queue position is unavailable."""
    status_obj = MagicMock()
    status_obj.name = "RUNNING"

    quantum_job = MagicMock(spec=QuantumJob)
    quantum_job.status.return_value = status_obj
    quantum_job.queue_position = MagicMock()
    quantum_job.queue_position.side_effect = Exception("Queue position not available")

    info = job_status(quantum_job)

    assert isinstance(info, JobStatusInfo)
    assert info.status == JobStatus.RUNNING
    assert info.queue_position is None
    assert info.queue_type is None
    assert info.message is None


def test_job_status_unknown():
    """Verify handling of unknown status."""
    quantum_job = MagicMock(spec=QuantumJob)
    quantum_job.status.side_effect = Exception("Status error")
    # Explicitly set queue_position to None to prevent auto-creation of nested mocks
    quantum_job.queue_position = None

    info = job_status(quantum_job)

    assert isinstance(info, JobStatusInfo)
    assert info.status == JobStatus.UNKNOWN
    assert info.queue_position is None
    assert info.queue_type is None
    assert info.message is None


def test_job_status_string_representation():
    """Verify string representation of JobStatusInfo."""
    # Basic status
    info = JobStatusInfo(status=JobStatus.RUNNING)
    assert str(info) == "RUNNING"

    # With queue position
    info = JobStatusInfo(status=JobStatus.QUEUED, queue_position=3)
    assert str(info) == "QUEUED (position 3)"

    # With queue type
    info = JobStatusInfo(
        status=JobStatus.QUEUED,
        queue_position=3,
        queue_type="PRIORITY"
    )
    assert str(info) == "QUEUED (position 3, PRIORITY)"

    # With message
    info = JobStatusInfo(
        status=JobStatus.FAILED,
        message="Circuit too large"
    )
    assert str(info) == "FAILED: Circuit too large"

    # Complete info
    info = JobStatusInfo(
        status=JobStatus.QUEUED,
        queue_position=3,
        queue_type="PRIORITY",
        message="High priority task"
    )
    assert str(info) == "QUEUED (position 3, PRIORITY): High priority task"
