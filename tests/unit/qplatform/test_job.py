from unittest.mock import MagicMock, patch
import pytest
from qbraid.runtime import QuantumJob, QiskitJob, AzureQuantumJob
from metriq_gym.qplatform.job import execution_time, job_status, JobStatusInfo, total_execution_time
from qbraid.runtime.quantinuum import QuantinuumJob
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


def test_execution_time_quantinuum():
    """Verify execution time is calculated correctly for QuantinuumJob."""
    start = datetime.now()
    completed = start + timedelta(seconds=15)

    mock_ref = MagicMock()
    mock_ref.last_status_detail = SimpleNamespace(
        running_time=start,
        completed_time=completed,
    )

    with (
        patch.object(QuantinuumJob, "_get_ref", return_value=mock_ref),
        patch.object(QuantinuumJob, "status", return_value=JobStatus.COMPLETED),
    ):
        quantinuum_job = QuantinuumJob(job_id="test-job-id")
        assert execution_time(quantinuum_job) == 15.0


def test_execution_time_quantinuum_not_completed():
    """Verify execution time raises ValueError when job is not completed."""
    with patch.object(QuantinuumJob, "status", return_value=JobStatus.RUNNING):
        quantinuum_job = QuantinuumJob(job_id="test-job-id")
        with pytest.raises(ValueError):
            execution_time(quantinuum_job)


def test_execution_time_quantinuum_missing_status_detail():
    """Verify execution time raises ValueError when last_status_detail is missing."""
    mock_ref = MagicMock()
    mock_ref.last_status_detail = None

    with (
        patch.object(QuantinuumJob, "_get_ref", return_value=mock_ref),
        patch.object(QuantinuumJob, "status", return_value=JobStatus.COMPLETED),
    ):
        quantinuum_job = QuantinuumJob(job_id="test-job-id")
        with pytest.raises(ValueError, match="last_status_detail is missing"):
            execution_time(quantinuum_job)


def test_execution_time_quantinuum_missing_timestamps():
    """Verify execution time raises ValueError when timestamps are missing."""
    mock_ref = MagicMock()
    mock_ref.last_status_detail = SimpleNamespace(
        running_time=datetime.now(),
        completed_time=None,  # missing completed_time
    )

    with (
        patch.object(QuantinuumJob, "_get_ref", return_value=mock_ref),
        patch.object(QuantinuumJob, "status", return_value=JobStatus.COMPLETED),
    ):
        quantinuum_job = QuantinuumJob(job_id="test-job-id")
        with pytest.raises(ValueError, match="completed_time or running_time is missing"):
            execution_time(quantinuum_job)


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


# --- failure_reason / failed_jobs_summary -----------------------------------------


def test_failure_reason_braket_uses_task_failure_reason():
    from qbraid.runtime import BraketQuantumTask
    from metriq_gym.qplatform.job import failure_reason

    task = object.__new__(BraketQuantumTask)
    task._task = SimpleNamespace(
        metadata=lambda: {"failureReason": "Error occurred during compilation"}
    )
    assert failure_reason(task) == "Error occurred during compilation"


def test_failure_reason_qiskit_uses_error_message():
    from metriq_gym.qplatform.job import failure_reason

    job = object.__new__(QiskitJob)
    job._job = SimpleNamespace(error_message=lambda: "Circuit too deep")
    assert failure_reason(job) == "Circuit too deep"


def test_failure_reason_azure_combines_code_and_message():
    from metriq_gym.qplatform.job import failure_reason

    job = object.__new__(AzureQuantumJob)
    job._job = SimpleNamespace(
        details=SimpleNamespace(error_data=SimpleNamespace(code="InvalidInput", message="bad"))
    )
    assert failure_reason(job) == "InvalidInput: bad"


def test_failure_reason_quantinuum_uses_last_message():
    from metriq_gym.qplatform.job import failure_reason

    mock_ref = MagicMock()
    mock_ref.last_message = "compile error"
    with patch.object(QuantinuumJob, "_get_ref", return_value=mock_ref):
        job = QuantinuumJob(job_id="test-job-id")
        assert failure_reason(job) == "compile error"


def test_failure_reason_never_raises_and_unknown_types_return_none():
    from metriq_gym.qplatform.job import failure_reason

    job = object.__new__(QiskitJob)
    job._job = SimpleNamespace(error_message=MagicMock(side_effect=RuntimeError("offline")))
    assert failure_reason(job) is None

    generic = MagicMock(spec=QuantumJob)
    assert failure_reason(generic) is None


def _qiskit_job_with(job_id: str, status: JobStatus, error_message=None) -> QiskitJob:
    job = object.__new__(QiskitJob)
    job._job_id = job_id  # backs the read-only ``id`` property
    job.status = lambda status=status: status
    job._job = SimpleNamespace(error_message=lambda: error_message)
    return job


def test_failed_jobs_summary_lists_only_terminal_failures():
    from metriq_gym.qplatform.job import failed_jobs_summary

    summary = failed_jobs_summary(
        [
            _qiskit_job_with("run-1", JobStatus.RUNNING),
            _qiskit_job_with("fail-1", JobStatus.FAILED, "Circuit too deep"),
            _qiskit_job_with("cancel-1", JobStatus.CANCELLED),
        ]
    )
    assert summary == "fail-1: FAILED - Circuit too deep\ncancel-1: CANCELLED"


def test_failed_jobs_summary_none_when_nothing_failed():
    from metriq_gym.qplatform.job import failed_jobs_summary

    assert failed_jobs_summary([_qiskit_job_with("q-1", JobStatus.QUEUED)]) is None
