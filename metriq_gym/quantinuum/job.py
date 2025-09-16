from typing import Any

import qnexus as qnx
from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result


class QuantinuumJob(QuantumJob):
    def __init__(self, job_id: str, *, device: Any | None = None, **_: Any) -> None:
        super().__init__(job_id, device)

    def _ref(self):
        return qnx.jobs.get(self.id)

    def result(self) -> Result:
        ref = self._ref()
        results = qnx.jobs.results(ref)
        print(results)
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
            return JobStatus.COMPLETED if qnx.jobs.results(self._ref()) else JobStatus.RUNNING
        except Exception:
            # Treat transient lookup issues as running to allow result path to proceed in fetch_result
            return JobStatus.RUNNING

    def cancel(self) -> bool:
        return False
