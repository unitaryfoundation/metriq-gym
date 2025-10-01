"""qBraid job wrapper for OriginQ QCloud executions."""

from __future__ import annotations

import logging
from typing import Any

from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result


logger = logging.getLogger(__name__)


def _ensure_pyqpanda3():
    try:
        from pyqpanda3 import qcloud  # noqa: F401  # pragma: nocover - import side effect only
    except ImportError as exc:  # pragma: no cover - import guard executed only when missing dep
        raise ImportError(
            "pyqpanda3 is required to use the Origin provider. Install it with 'pip install pyqpanda3'."
        ) from exc


def _map_status(origin_status: Any) -> JobStatus:
    _ensure_pyqpanda3()
    from pyqpanda3.qcloud import JobStatus as OriginJobStatus

    if origin_status == OriginJobStatus.FINISHED:
        return JobStatus.COMPLETED
    if origin_status in (OriginJobStatus.WAITING, OriginJobStatus.QUEUING):
        return JobStatus.QUEUED
    if origin_status == OriginJobStatus.COMPUTING:
        return JobStatus.RUNNING
    if origin_status == OriginJobStatus.FAILED:
        return JobStatus.FAILED
    return JobStatus.UNKNOWN


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

        _ensure_pyqpanda3()
        from pyqpanda3.qcloud import QCloudJob

        try:
            self._backend_job = QCloudJob(self.id)
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
