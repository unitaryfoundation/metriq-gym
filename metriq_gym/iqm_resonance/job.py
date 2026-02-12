from __future__ import annotations

from typing import Any
import logging

from iqm.iqm_client import Status
from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result

logger = logging.getLogger(__name__)


class IQMResonanceJob(QuantumJob):
    def __init__(
        self,
        job_id: str,
        *,
        device: Any | None = None,
        client=None,
        measurement_keys: list[list[str]] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(job_id, device)
        self._client = client
        self._measurement_keys = measurement_keys or []

    def _counts_from_measurements(
        self, measurement: dict[str, list[list[int]]], keys: list[str] | None
    ) -> dict[str, int]:
        if not measurement:
            return {}

        first_val = next(iter(measurement.values()))
        num_shots = len(first_val)
        counts: dict[str, int] = {}
        ordered_keys = keys or list(measurement.keys())

        for shot_idx in range(num_shots):
            bits: list[str] = []
            for key in ordered_keys:
                val = measurement.get(key, [])
                try:
                    bit_list = val[shot_idx]
                except Exception:
                    bit_list = []
                bits.extend(str(int(b)) for b in bit_list)
            bitstring = "".join(bits) if bits else "0"
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def result(self) -> Result:
        if self._client is None:
            raise RuntimeError("iqm-client not configured on job")

        run_result = self._client.wait_for_results(self.id)
        measurements = run_result.measurements or []
        if not measurements:
            raise RuntimeError(f"No results available for job {self.id}")

        counts: dict[str, int] = {}
        for idx, meas in enumerate(measurements):
            keys = None
            if 0 <= idx < len(self._measurement_keys):
                keys = self._measurement_keys[idx]
            partial = self._counts_from_measurements(meas, keys)
            for bitstring, val in partial.items():
                counts[bitstring] = counts.get(bitstring, 0) + val
        return Result(
            device_id=self._device.name if self._device else "iqm",
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=counts),
        )

    def status(self) -> JobStatus:
        if self._client is None:
            return JobStatus.UNKNOWN

        try:
            status_obj = self._client.get_run_status(self.id)
            status_val = status_obj.status if hasattr(status_obj, "status") else status_obj
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to poll IQM job %s: %s", self.id, exc)
            return JobStatus.UNKNOWN

        if status_val == Status.READY:
            return JobStatus.COMPLETED
        if status_val in {Status.PENDING_COMPILATION, Status.PENDING_EXECUTION}:
            return JobStatus.QUEUED
        if status_val == Status.FAILED:
            return JobStatus.FAILED
        if status_val == Status.ABORTED:
            return JobStatus.CANCELLED
        return JobStatus.UNKNOWN

    def cancel(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.abort_job(self.id)
            return True
        except Exception:
            return False
