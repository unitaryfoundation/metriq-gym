import datetime as dt

from qbraid import QuantumDevice
from qbraid.runtime import (
    QuantumJob,
    JobStatus,
    Result,
    GateModelResultData,
)
from qbraid_core import ResourceNotFoundError

from ._store import write, read


class LocalAerJob(QuantumJob):
    def __init__(
        self,
        job_id: str,
        *,
        device: QuantumDevice | None = None,
        counts: dict[str, int] | None = None,
        execution_time: float | None = None,
        **_,
    ) -> None:
        super().__init__(job_id, device)
        self._execution_time_s = execution_time

        if counts is not None:
            if device is None:
                raise ValueError("device must be provided")

            self._counts = counts
            self._device_id = device.id

            write(
                job_id,
                {
                    "job_id": job_id,
                    "device_id": self._device_id,
                    "completed_at_utc": dt.datetime.now().isoformat(),
                    "counts": counts,
                    "execution_time_s": execution_time,
                },
            )
        else:  # At polling time a job is created with load_job
            data = read(job_id)
            if data is None:
                raise ResourceNotFoundError(f"Job {job_id!r} not found")

            self._counts = data.get("counts", {})
            self._device_id = data.get("device_id")
            self._execution_time_s = data.get("execution_time_s")

    def result(self) -> Result:
        return Result(
            device_id=self._device_id,
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=self._counts),
        )

    def status(self) -> JobStatus:
        return JobStatus.COMPLETED

    def cancel(self) -> bool:
        return False
