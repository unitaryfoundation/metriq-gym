from typing import Any

import qnexus as qnx
from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result


class QuantinuumJob(QuantumJob):
    def __init__(self, job_id: str, *, device: Any | None = None, **_: Any) -> None:
        super().__init__(job_id, device)
        self._ref_obj = None

    def _get_ref(self):
        # Lazily resolve and cache the qnexus job reference; fall back to id when unavailable
        if self._ref_obj is not None:
            return self._ref_obj
        try:
            self._ref_obj = qnx.jobs.get(self.id)
            return self._ref_obj
        except Exception:
            return self.id

    def result(self) -> Result:
        ref = self._get_ref()
        results = qnx.jobs.results(ref)
        if not results:
            # TODO: don't make it blocking
            qnx.jobs.wait_for(ref)
            results = qnx.jobs.results(ref)
        if not results:
            raise RuntimeError(f"No results available for job {self.id}")

        counts = results[0].download_result().get_counts()
        norm_counts = {"".join(map(str, k)): v for k, v in counts.items()}

        return Result(
            device_id=self._device.name if self._device else "unknown",
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=norm_counts),
        )

    def status(self) -> JobStatus:
        try:
            return JobStatus.COMPLETED if qnx.jobs.results(self._get_ref()) else JobStatus.RUNNING
        except Exception:
            # Treat transient lookup issues as running to allow result path to proceed in fetch_result
            return JobStatus.RUNNING

    def cancel(self) -> bool:
        try:
            qnx.jobs.cancel(self._get_ref())
            return True
        except Exception:
            return False
