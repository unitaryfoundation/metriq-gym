from dataclasses import dataclass
from functools import singledispatch
from typing import Iterable

from qbraid import QuantumJob
from qbraid.runtime import QiskitJob, AzureQuantumJob, BraketQuantumTask
from qbraid.runtime.enums import JobStatus
from qiskit_ibm_runtime.execution_span import ExecutionSpans

from metriq_gym.local.job import LocalAerJob
from qbraid.runtime.quantinuum import QuantinuumJob
from qbraid.runtime.quantinuum.job import QuantinuumJobError


@singledispatch
def execution_time(quantum_job: QuantumJob) -> float:
    raise NotImplementedError(f"Execution time not implemented for type {type(quantum_job)}")


@execution_time.register
def _(quantum_job: QiskitJob) -> float:
    execution_spans: ExecutionSpans = quantum_job._job.result().metadata["execution"][
        "execution_spans"
    ]
    return (execution_spans.stop - execution_spans.start).total_seconds()


@execution_time.register
def _(quantum_job: AzureQuantumJob) -> float:
    start_time = quantum_job._job.details.begin_execution_time
    end_time = quantum_job._job.details.end_execution_time
    if start_time is None or end_time is None:
        raise ValueError("Execution time not available")
    return (end_time - start_time).total_seconds()


@execution_time.register
def _(quantum_job: LocalAerJob) -> float:
    if quantum_job._execution_time_s is None:
        raise ValueError("Execution time not available")
    return quantum_job._execution_time_s


@execution_time.register
def _(quantum_job: QuantinuumJob) -> float:
    try:
        res = quantum_job.execution_time_s()
    except QuantinuumJobError as exc:
        raise ValueError(str(exc)) from exc
    if res is None:
        raise ValueError("Execution time not available")
    return res


def total_execution_time(quantum_jobs: Iterable[QuantumJob]) -> float | None:
    """Sum execution time for completed jobs, skipping jobs that do not report it."""
    total = None
    for qjob in quantum_jobs:
        if qjob.status() != JobStatus.COMPLETED:
            continue
        try:
            t = execution_time(qjob)
        except (NotImplementedError, ValueError):
            continue
        total = t if total is None else total + t
    return total


FAILED_STATUSES = frozenset({JobStatus.FAILED, JobStatus.CANCELLED})


@singledispatch
def failure_reason(quantum_job: QuantumJob) -> str | None:
    """Best-effort, provider-agnostic error message for a failed job.

    Returns None when the provider exposes no failure detail. Never raises: this is
    called while recording a failure, and a broken accessor must not mask it.
    """
    return None


@failure_reason.register
def _(quantum_job: BraketQuantumTask) -> str | None:
    try:
        reason = quantum_job._task.metadata().get("failureReason")
    except Exception:
        return None
    return str(reason) if reason else None


@failure_reason.register
def _(quantum_job: QiskitJob) -> str | None:
    try:
        reason = quantum_job._job.error_message()
    except Exception:
        return None
    return str(reason) if reason else None


@failure_reason.register
def _(quantum_job: AzureQuantumJob) -> str | None:
    try:
        error_data = quantum_job._job.details.error_data
    except Exception:
        return None
    if error_data is None:
        return None
    code = getattr(error_data, "code", None)
    message = getattr(error_data, "message", None)
    if code and message:
        return f"{code}: {message}"
    return str(message or code) if (message or code) else None


@failure_reason.register
def _(quantum_job: QuantinuumJob) -> str | None:
    try:
        reason = quantum_job._get_ref().last_message
    except Exception:
        return None
    return str(reason) if reason else None


def failed_jobs_summary(quantum_jobs: Iterable[QuantumJob]) -> str | None:
    """Describe every task in a terminal failed/cancelled state, or None if there is none.

    The summary is one line per failed task (``<id>: <STATUS> - <reason>``) so that it
    can be stored verbatim as evidence on the job record.
    """
    lines: list[str] = []
    for qjob in quantum_jobs:
        try:
            status = qjob.status()
        except Exception:
            continue
        if status not in FAILED_STATUSES:
            continue
        line = f"{qjob.id}: {status.value}"
        reason = failure_reason(qjob)
        if reason:
            line += f" - {reason}"
        lines.append(line)
    return "\n".join(lines) if lines else None


@dataclass
class JobStatusInfo:
    """Provider agnostic job status information."""

    status: JobStatus
    queue_position: int | None = None


def extract_status_info(quantum_job: QuantumJob, supports_queue_position: bool) -> JobStatusInfo:
    """Helper to extract job status and optionally queue position."""
    try:
        status_obj = quantum_job.status()
        raw_status = getattr(status_obj, "name", str(status_obj)).upper()
        status = JobStatus(raw_status) if raw_status in JobStatus.__members__ else JobStatus.UNKNOWN
    except Exception:
        status = JobStatus.UNKNOWN

    queue_position = None
    if supports_queue_position:
        for attr in ["queue_position", "queue_info"]:
            # These attributes are defined in qBraid provider job classes (e.g., QiskitJob, BraketQuantumTask).
            # Reference: https://github.com/qBraid/qBraid/tree/main/qbraid/runtime
            if hasattr(quantum_job, attr):
                try:
                    info = getattr(quantum_job, attr)
                    info = info() if callable(info) else info
                    if hasattr(info, "position"):
                        info = info.position
                    if info is not None:
                        queue_position = int(info)
                    break
                except Exception:
                    continue

    return JobStatusInfo(status=status, queue_position=queue_position)


@singledispatch
def job_status(quantum_job: QuantumJob) -> JobStatusInfo:
    """Fallback for unknown provider types: status only."""
    return extract_status_info(quantum_job, supports_queue_position=False)


@job_status.register
def _(quantum_job: QiskitJob) -> JobStatusInfo:
    return extract_status_info(quantum_job, supports_queue_position=True)


@job_status.register
def _(quantum_job: BraketQuantumTask) -> JobStatusInfo:
    return extract_status_info(quantum_job, supports_queue_position=True)


@job_status.register
def _(quantum_job: AzureQuantumJob) -> JobStatusInfo:
    return extract_status_info(quantum_job, supports_queue_position=False)
