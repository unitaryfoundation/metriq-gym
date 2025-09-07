from __future__ import annotations
from typing import Any

from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result
from qbraid_core import ResourceNotFoundError
from metriq_gym.local._store import read
from .auth import load_api


class QuantinuumJob(QuantumJob):
    """Quantinuum job wrapper.

    Wraps a Quantinuum remote job. Uses pytket-quantinuum to fetch
    status and results using environment-provided credentials.
    """

    def __init__(self, job_id: str, *, device=None, **_: Any) -> None:
        super().__init__(job_id, device)
        self._device_id = device.id if device is not None else None
        # Try local store first (synchronous emulator path)
        data = read(job_id)
        if data is not None:
            self._counts = data.get("counts")
        else:
            self._counts = None

    def result(self) -> Result:
        # Lazily fetch results from Quantinuum when first requested
        if self._counts is None:
            try:
                from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency not installed
                raise RuntimeError(
                    "Missing dependency: pytket-quantinuum. Install with: poetry add pytket-quantinuum."
                ) from exc

            api = load_api()

            if self._device_id is None:
                raise ResourceNotFoundError("Device id missing for Quantinuum job")

            try:
                backend = QuantinuumBackend(self._device_id, api_handler=api)
                # Fetch result; API varies by version. Try convenient access patterns.
                result = backend.get_result(self.id)  # type: ignore[attr-defined]
                # Attempt to extract counts; adjust if API differs
                counts = getattr(result, "get_counts", None)
                if callable(counts):
                    self._counts = counts()  # type: ignore[assignment]
                else:
                    # Fallback structure
                    self._counts = getattr(result, "counts", None)
                if self._counts is None:
                    raise RuntimeError("Unable to extract measurement counts from Quantinuum result")
            except Exception as exc:
                raise RuntimeError("Failed to retrieve results for Quantinuum job") from exc

        return Result(
            device_id=self._device_id,
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=self._counts),
        )

    def status(self) -> JobStatus:
        # Try to query remote status; if unavailable, fall back to UNKNOWN
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
            api = load_api()

            if self._device_id is None:
                return JobStatus.UNKNOWN

            backend = QuantinuumBackend(self._device_id, api_handler=api)
            # Map backend-specific job status to qBraid JobStatus
            q_status = getattr(backend, "get_job_status", None)
            if callable(q_status):
                status_str = str(q_status(self.id)).lower()  # type: ignore[arg-type]
                if "completed" in status_str or "finished" in status_str:
                    return JobStatus.COMPLETED
                if "running" in status_str or "submitted" in status_str:
                    return JobStatus.RUNNING
                if "queued" in status_str or "pending" in status_str:
                    return JobStatus.QUEUED
                if "cancelled" in status_str or "canceled" in status_str:
                    return JobStatus.CANCELLED
                if "failed" in status_str or "error" in status_str:
                    return JobStatus.ERROR
            return JobStatus.UNKNOWN
        except Exception:
            return JobStatus.UNKNOWN

    def cancel(self) -> bool:
        # Implement cancel if the backend supports it; otherwise, return False.
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
            api = load_api()

            if self._device_id is None:
                return False

            backend = QuantinuumBackend(self._device_id, api_handler=api)
            cancel_fn = getattr(backend, "cancel_job", None)
            if callable(cancel_fn):
                return bool(cancel_fn(self.id))  # type: ignore[arg-type]
            return False
        except Exception:
            return False
