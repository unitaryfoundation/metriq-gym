from unittest.mock import MagicMock
import pytest
from qbraid.runtime import QuantumJob, QiskitJob, AzureQuantumJob
from metriq_gym.qplatform.job import execution_time, job_status, JobStatusInfo, total_execution_time
from qbraid.runtime.enums import JobStatus
from datetime import datetime, timedelta
from types import SimpleNamespace


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
    assert info.status == JobStatus.QUEUED
    assert info.queue_position == 3


def test_job_status_without_queue_position():
    """Verify fallback when queue position is unavailable."""
    status_obj = MagicMock()
    status_obj.name = "RUNNING"

    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job.status.return_value = status_obj
    # simulate method absence
    if hasattr(qiskit_job, "queue_position"):
        del qiskit_job.queue_position

    info = job_status(qiskit_job)

    assert isinstance(info, JobStatusInfo)
    assert info.status == JobStatus.RUNNING
    assert info.queue_position is None


def test_job_status_unknown_fallback():
    """Verify fallback to JobStatus.UNKNOWN for unrecognized statuses."""
    status_obj = MagicMock()
    status_obj.name = "FOOBAR"  # not a valid JobStatus

    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job.status.return_value = status_obj

    info = job_status(qiskit_job)

    assert isinstance(info, JobStatusInfo)
    assert info.status == JobStatus.UNKNOWN


def make_qiskit_job(duration_seconds: float, status: JobStatus) -> QiskitJob:
    start = datetime.now()
    spans = SimpleNamespace(start=start, stop=start + timedelta(seconds=duration_seconds))
    job = object.__new__(QiskitJob)
    job.status = lambda status=status: status
    job._job = SimpleNamespace(
        result=lambda: SimpleNamespace(metadata={"execution": {"execution_spans": spans}})
    )
    return job


def test_total_execution_time_sums_completed_jobs():
    job_pending = make_qiskit_job(duration_seconds=0, status=JobStatus.RUNNING)
    job_fast = make_qiskit_job(duration_seconds=5.0, status=JobStatus.COMPLETED)
    job_slow = make_qiskit_job(duration_seconds=7.5, status=JobStatus.COMPLETED)

    result = total_execution_time([job_pending, job_fast, job_slow])

    assert result == pytest.approx(12.5)


def test_total_execution_time_skips_unreported():
    job_not_impl = MagicMock(spec=QuantumJob)
    job_not_impl.status.return_value = JobStatus.COMPLETED

    job_value_error = object.__new__(AzureQuantumJob)
    job_value_error.status = lambda: JobStatus.COMPLETED
    job_value_error._job = SimpleNamespace(
        details=SimpleNamespace(begin_execution_time=None, end_execution_time=None)
    )

    job_valid = make_qiskit_job(4.2, status=JobStatus.COMPLETED)

    result = total_execution_time([job_not_impl, job_value_error, job_valid])

    assert result == pytest.approx(4.2)
