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


def extract_status_info(quantum_job: QuantumJob, supports_queue_position: bool) -> JobStatusInfo:
    """Helper to extract job status and optionally queue position."""
    try:
        status_obj = quantum_job.status()
        status = getattr(status_obj, "name", str(status_obj))
    except Exception:
        status = "UNKNOWN"

    queue_position = None
    if supports_queue_position:
        for attr in ["queue_position", "queue_info"]:
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
