from typing import Any

import qnexus as qnx
from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result


class QuantinuumJob(QuantumJob):
    def __init__(self, job_id: str, *, device: Any | None = None, **_: Any) -> None:
        super().__init__(job_id, device)

    def result(self) -> Result:
        results = qnx.jobs.results(self.id)
        if not results:
            qnx.jobs.wait_for(self.id)
            results = qnx.jobs.results(self.id)
        if not results:
            raise RuntimeError(f"No results available for job {self.id}")

        item0 = results[0]
        counts = None
        if hasattr(item0, "download_result"):
            backend_result = item0.download_result()
            if hasattr(backend_result, "get_counts"):
                counts = backend_result.get_counts()
        if counts is None and hasattr(item0, "download_counts"):
            counts = item0.download_counts()
        if counts is None and hasattr(item0, "get_output"):
            out = item0.get_output()
            if hasattr(out, "get_counts"):
                counts = out.get_counts()
        if counts is None:
            raise RuntimeError("Result object does not provide counts")

        return Result(
            device_id="quantinuum",
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=counts),
        )

    def status(self) -> JobStatus:
        try:
            return JobStatus.COMPLETED if qnx.jobs.results(self.id) else JobStatus.RUNNING
        except Exception:
            return JobStatus.UNKNOWN

    def cancel(self) -> bool:
        return False
