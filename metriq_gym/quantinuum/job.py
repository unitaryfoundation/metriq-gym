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

        # Collect counts from all results (handles batch job submissions)
        all_counts = []
        for result in results:
            counts = result.download_result().get_counts()
            norm_counts = {"".join(map(str, k)): v for k, v in counts.items()}
            all_counts.append(norm_counts)

        # Return single counts dict if only one result, otherwise return list
        measurement_counts = all_counts[0] if len(all_counts) == 1 else all_counts

        return Result(
            device_id=self._device.name if self._device else "unknown",
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=measurement_counts),
        )

    def status(self) -> JobStatus:
        try:
            ref = self._get_ref()
            last_status = getattr(ref, "last_status", None)
            match last_status:
                case "COMPLETED":
                    return JobStatus.COMPLETED
                case "ERROR":
                    last_message = getattr(ref, "last_message", None)
                    print(f"Job failed with: \n {last_message}")
                    return JobStatus.FAILED
                case "QUEUED" | "RUNNING" | "INITIALIZING":
                    return JobStatus.RUNNING
                case _:
                    return JobStatus.UNKNOWN
        except Exception:
            return JobStatus.UNKNOWN

    def cancel(self) -> bool:
        try:
            qnx.jobs.cancel(self._get_ref())
            return True
        except Exception:
            return False
