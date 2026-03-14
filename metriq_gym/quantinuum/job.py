from typing import Any

import qnexus as qnx
from pytket.circuit import BasisOrder
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
            self._ref_obj = qnx.jobs.get(id=self.id)
            return self._ref_obj
        except Exception:
            return self.id

    def execution_time_s(self) -> float | None:
        """
        Return the wall-clock time for the job execution in seconds, as reported by the backend.

        For Quantinuum jobs, this is computed as the backend's
        ``last_status_detail.completed_time - last_status_detail.running_time``.
        This duration reflects the total time the job spent in the "running" phase on the service
        and may include queueing/wait times, calibration operations, and other backend checks,
        not just the time during which the quantum device was actively executing the circuit.

        A more granular, on-device-only execution time metric is not exposed by the current
        Quantinuum/qnexus API, so this is the most precise execution-time estimate available.

        Returns:
            The execution time in seconds, or None if the job is not completed.

        Raises:
            ValueError: If the job is completed but execution time details are unavailable.
        """
        if self.status() != JobStatus.COMPLETED:
            return None
        ref = self._get_ref()
        last_status_detail = getattr(ref, "last_status_detail", None)
        if last_status_detail is None:
            raise ValueError("Execution time not available: last_status_detail is missing")
        completed_time = getattr(last_status_detail, "completed_time", None)
        running_time = getattr(last_status_detail, "running_time", None)
        if completed_time is None or running_time is None:
            raise ValueError(
                "Execution time not available: completed_time or running_time is missing"
            )
        return (completed_time - running_time).total_seconds()

    def result(self) -> Result:
        ref = self._get_ref()
        results = qnx.jobs.results(ref)
        if not results:
            # TODO: don't make it blocking
            qnx.jobs.wait_for(ref)
            results = qnx.jobs.results(ref)
        if not results:
            raise RuntimeError(f"No results available for job {self.id}")

        all_counts = []
        for result in results:
            # Quantinuum (as documented in pytket) by default uses bitstrings
            # with the least significant bit first. We convert to
            # most significant bit first (dlo = descending lexographic order)
            # for consistency with other backends.
            counts = result.download_result().get_counts(basis=BasisOrder.dlo)
            norm_counts = {"".join(map(str, k)): v for k, v in counts.items()}
            all_counts.append(norm_counts)

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
