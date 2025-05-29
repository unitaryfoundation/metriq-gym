from dataclasses import dataclass
from functools import singledispatch

from qbraid import QuantumJob
from qbraid.runtime import QiskitJob, AzureQuantumJob, BraketQuantumTask
from qiskit_ibm_runtime.execution_span import ExecutionSpans


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
def _(quantum_job: BraketQuantumTask) -> float:
    # TODO: for speed benchmarking, we need 'execution' metadata instead of 'createdAt' and 'endedAt'
    return (
        quantum_job._task.metadata()["endedAt"] - quantum_job._task.metadata()["createdAt"]
    ).total_seconds()

@dataclass
class JobStatusInfo:
    """Provider agnostic job status information."""

    status: str
    queue_position: int | None = None


@singledispatch
def job_status(quantum_job: QuantumJob) -> JobStatusInfo:
    """Return the provider specific status and queue position if available."""
    try:
        status_obj = quantum_job.status()
        status = getattr(status_obj, "name", str(status_obj))
    except Exception:
        status = "UNKNOWN"
    queue_position = None
    for attr in ["queue_position", "queue_info", "queue_position_retrieval"]:
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
                pass
    return JobStatusInfo(status=status, queue_position=queue_position)


@job_status.register
def _(quantum_job: QiskitJob) -> JobStatusInfo:
    status = getattr(quantum_job._job.status(), "name", str(quantum_job._job.status()))
    queue_position = None
    if hasattr(quantum_job._job, "queue_position"):
        try:
            queue_position = quantum_job._job.queue_position()
        except Exception:
            queue_position = None
    elif hasattr(quantum_job._job, "queue_info"):
        try:
            info = quantum_job._job.queue_info()
            if info:
                queue_position = info.position
        except Exception:
            queue_position = None
    return JobStatusInfo(status=status, queue_position=queue_position)


@job_status.register
def _(quantum_job: AzureQuantumJob) -> JobStatusInfo:
    status_obj = getattr(quantum_job._job.details, "status", None)
    status = getattr(status_obj, "name", str(status_obj)) if status_obj else "UNKNOWN"
    queue_position = getattr(quantum_job._job.details, "current_queue_position", None)
    return JobStatusInfo(status=status, queue_position=queue_position)


@job_status.register
def _(quantum_job: BraketQuantumTask) -> JobStatusInfo:
    status_obj = None
    try:
        status_obj = quantum_job._task.state()
    except Exception:
        pass
    status = getattr(status_obj, "name", str(status_obj)) if status_obj else "UNKNOWN"
    queue_position = getattr(quantum_job._task, "queue_position", None)
    return JobStatusInfo(status=status, queue_position=queue_position)

