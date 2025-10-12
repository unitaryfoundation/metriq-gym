"""qBraid job wrapper for OriginQ QCloud executions."""

import logging
from typing import Any

from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result

from .qcloud_utils import get_qcloud_job


logger = logging.getLogger(__name__)


_STATUS_LOOKUP = {
    "FINISH": JobStatus.COMPLETED,
    "FINISHED": JobStatus.COMPLETED,
    "COMPLETED": JobStatus.COMPLETED,
    "WAITING": JobStatus.QUEUED,
    "QUEUING": JobStatus.QUEUED,
    "QUEUED": JobStatus.QUEUED,
    "COMPUTING": JobStatus.RUNNING,
    "RUNNING": JobStatus.RUNNING,
    "EXECUTING": JobStatus.RUNNING,
    "FAILED": JobStatus.FAILED,
    "ERROR": JobStatus.FAILED,
}


def _normalize_status(origin_status: Any) -> str:
    """Convert various SDK status representations to a comparable uppercase string."""
    candidate: Any
    if hasattr(origin_status, "name"):
        candidate = getattr(origin_status, "name")
    elif hasattr(origin_status, "value"):
        candidate = getattr(origin_status, "value")
    else:
        candidate = origin_status

    normalized = str(candidate).strip().upper()
    if normalized.startswith("JOBSTATUS."):
        normalized = normalized.split(".", 1)[1]
    if normalized.startswith("JOB_"):
        normalized = normalized[4:]
    return normalized


def _map_status(origin_status: Any) -> JobStatus:
    normalized = _normalize_status(origin_status)
    return _STATUS_LOOKUP.get(normalized, JobStatus.UNKNOWN)


class OriginJob(QuantumJob):
    """Represents a single QCloud execution."""

    def __init__(
        self,
        job_id: str,
        *,
        device: Any | None = None,
        backend_job: Any | None = None,
        **_: Any,
    ) -> None:
        super().__init__(job_id, device)
        self._backend_job = backend_job

    def _get_backend_job(self):
        if self._backend_job is not None:
            return self._backend_job
        try:
            self._backend_job = get_qcloud_job(self.id)
        except Exception as exc:  # pragma: no cover - depends on live service
            logger.error("Failed to retrieve OriginQ job %s", self.id, exc_info=True)
            raise RuntimeError(f"Unable to retrieve OriginQ job {self.id}") from exc
        return self._backend_job

    def result(self) -> Result:
        backend_job = self._get_backend_job()

        try:
            result_obj = backend_job.result()
        except Exception as exc:  # pragma: no cover - depends on live service
            logger.error("Failed to fetch OriginQ job result %s", self.id, exc_info=True)
            raise RuntimeError(f"Failed to fetch results for OriginQ job {self.id}") from exc

        counts = result_obj.get_counts()
        if not counts:
            try:
                counts_list = result_obj.get_counts_list()
            except Exception:  # pragma: no cover - depends on live service
                counts_list = []
            if counts_list:
                counts = counts_list[0]

        normalized = {str(key): int(value) for key, value in counts.items()}

        return Result(
            device_id=self._device.profile.device_id if self._device else "origin",
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=normalized),
        )

    def status(self) -> JobStatus:
        backend_job = self._get_backend_job()
        try:
            status = backend_job.status()
        except Exception:  # pragma: no cover - depends on live service
            logger.debug("Unable to retrieve job status for %s", self.id, exc_info=True)
            return JobStatus.UNKNOWN
        return _map_status(status)

    def cancel(self) -> bool:
        # QCloud does not currently expose cancellation through the Python SDK.
        return False
