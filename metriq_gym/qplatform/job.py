from dataclasses import dataclass
from enum import StrEnum
from functools import singledispatch
from typing import Optional, Any

from qbraid import QuantumJob
from qbraid.runtime import QiskitJob, AzureQuantumJob, BraketQuantumTask
from qbraid.runtime.enums import JobStatus as QbraidJobStatus
from qiskit_ibm_runtime.execution_span import ExecutionSpans


class JobStatus(StrEnum):
    """Provider-agnostic job status enum.
    
    This enum extends qBraid's JobStatus with additional statuses and
    provides a consistent interface across different providers.
    """
    INITIALIZING = "INITIALIZING"
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    HOLD = "HOLD"

    @classmethod
    def from_qbraid(cls, status: QbraidJobStatus) -> "JobStatus":
        """Convert qBraid JobStatus to metriq-gym JobStatus."""
        return cls(status.name)

    @classmethod
    def terminal_states(cls) -> set["JobStatus"]:
        """Returns the final job statuses."""
        return {cls.COMPLETED, cls.CANCELLED, cls.FAILED}


@dataclass
class JobStatusInfo:
    """Provider-agnostic job status information.
    
    This class provides a consistent interface for job status information
    across different quantum computing providers.
    
    Attributes:
        status: The current status of the job
        queue_position: Optional queue position if available from provider
        queue_type: Optional queue type (e.g. "NORMAL", "PRIORITY") if available
        message: Optional status message or error details
    """
    status: JobStatus
    queue_position: Optional[int] = None
    queue_type: Optional[str] = None
    message: Optional[str] = None

    def __str__(self) -> str:
        """String representation of job status info."""
        msg = f"{self.status.value}"
        if self.queue_position is not None:
            msg += f" (position {self.queue_position}"
            if self.queue_type:
                msg += f", {self.queue_type}"
            msg += ")"
        if self.message:
            msg += f": {self.message}"
        return msg


def extract_status_info(quantum_job: QuantumJob) -> JobStatusInfo:
    """Extract job status information from a quantum job.
    
    This is a helper function that handles the common case of extracting
    status information from a quantum job. It uses the provider-specific
    implementations where available.
    
    Args:
        quantum_job: The quantum job to extract status from
        
    Returns:
        JobStatusInfo containing the extracted status information
    """
    try:
        status = JobStatus.from_qbraid(quantum_job.status())
    except Exception:
        status = JobStatus.UNKNOWN

    queue_position = None
    queue_type = None
    message = None

    # Try to get queue position if available
    if hasattr(quantum_job, "queue_position"):
        try:
            queue_info = quantum_job.queue_position()
            if isinstance(queue_info, (int, str)):
                queue_position = int(queue_info)
            elif hasattr(queue_info, "position"):
                queue_position = queue_info.position
            if hasattr(queue_info, "queue_type"):
                queue_type = queue_info.queue_type
            if hasattr(queue_info, "message"):
                message = queue_info.message
        except Exception:
            pass

    return JobStatusInfo(
        status=status,
        queue_position=queue_position,
        queue_type=queue_type,
        message=message
    )


@singledispatch
def job_status(quantum_job: QuantumJob) -> JobStatusInfo:
    """Get status information for a quantum job.
    
    This is the default implementation that works for any QuantumJob.
    Provider-specific implementations can be registered for better handling.
    
    Args:
        quantum_job: The quantum job to get status for
        
    Returns:
        JobStatusInfo containing the job's status information
    """
    return extract_status_info(quantum_job)


@job_status.register
def _(quantum_job: QiskitJob) -> JobStatusInfo:
    """Get status information for an IBM Qiskit job.
    
    IBM jobs provide queue position information through the queue_position()
    method. The position is within the scope of the provider.
    """
    return extract_status_info(quantum_job)


@job_status.register
def _(quantum_job: BraketQuantumTask) -> JobStatusInfo:
    """Get status information for an AWS Braket quantum task.
    
    AWS Braket tasks provide queue position and type information through
    the queue_position() method. The queue type can be "NORMAL" or "PRIORITY".
    """
    return extract_status_info(quantum_job)


@job_status.register
def _(quantum_job: AzureQuantumJob) -> JobStatusInfo:
    """Get status information for an Azure Quantum job.
    
    Azure Quantum jobs currently don't provide queue position information.
    """
    return extract_status_info(quantum_job)


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
