from __future__ import annotations

from typing import Any

from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result


def _get_job_ref(job_id: str):  # pragma: no cover - requires qnexus
    import qnexus as qnx  # type: ignore

    # Best-effort reconstruction of a JobRef from the ID
    # Prefer explicit getter if available; else create a lightweight proxy
    try:
        return qnx.jobs.get(job_id)
    except Exception:
        class _JobProxy:
            def __init__(self, _id: str):
                self.id = _id

        return _JobProxy(job_id)


class QuantinuumJob(QuantumJob):
    def __init__(self, job_id: str, *, device: Any | None = None, **_: Any) -> None:
        super().__init__(job_id, device)

    def result(self) -> Result:  # pragma: no cover - requires qnexus
        import qnexus as qnx  # type: ignore
        try:
            from qnexus.exceptions import AuthenticationError  # type: ignore
        except Exception:
            class AuthenticationError(Exception):
                pass

        from .utils import ensure_login

        job_ref = _get_job_ref(self.id)
        # Block until completion; the CLI poller calls this when status is complete
        try:
            qnx.jobs.wait_for(job_ref)
        except AuthenticationError:
            ensure_login()
            qnx.jobs.wait_for(job_ref)
        try:
            results = qnx.jobs.results(job_ref)
        except AuthenticationError:
            ensure_login()
            results = qnx.jobs.results(job_ref)
        if not results:
            raise RuntimeError(f"No results available for job {self.id}")
        # Download pytket BackendResult and convert to counts
        backend_result = results[0].download_result()
        counts = backend_result.get_counts()

        return Result(
            device_id=self.device.id if self.device else "quantinuum",
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=counts),
        )

    def status(self) -> JobStatus:  # pragma: no cover - requires qnexus
        try:
            import qnexus as qnx  # type: ignore
            try:
                from qnexus.exceptions import AuthenticationError  # type: ignore
            except Exception:
                class AuthenticationError(Exception):
                    pass

            job_ref = _get_job_ref(self.id)
            # Many SDKs provide .status or a details endpoint; try both
            status = getattr(job_ref, "status", None)
            if callable(status):
                try:
                    status = status()
                except AuthenticationError:
                    from .utils import ensure_login

                    ensure_login()
                    status = status()
            if not status:
                try:
                    # pandas dataframe view
                    df = job_ref.df()
                    if hasattr(df, "status"):
                        status = str(df.status.iloc[0])
                except Exception:
                    pass
            if not status:
                # If result is ready, consider completed
                try:
                    res = qnx.jobs.results(job_ref)
                except AuthenticationError:
                    from .utils import ensure_login

                    ensure_login()
                    res = qnx.jobs.results(job_ref)
                if res:
                    return JobStatus.COMPLETED
                return JobStatus.RUNNING

            status_str = str(status).upper()
            if "COMPLETE" in status_str or status_str == "COMPLETED":
                return JobStatus.COMPLETED
            if "RUN" in status_str or status_str == "QUEUED" or status_str == "PENDING":
                return JobStatus.RUNNING
            if "CANCEL" in status_str:
                return JobStatus.CANCELLED
            if "FAIL" in status_str or "ERROR" in status_str:
                return JobStatus.ERROR
            return JobStatus.UNKNOWN
        except Exception:
            return JobStatus.UNKNOWN

    def cancel(self) -> bool:  # pragma: no cover - requires qnexus
        try:
            job_ref = _get_job_ref(self.id)
            cancel = getattr(job_ref, "cancel", None)
            if callable(cancel):
                cancel()
                return True
        except Exception:
            pass
        return False
