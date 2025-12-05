import logging
from typing import Any

import qnexus as qnx
from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result

logger = logging.getLogger(__name__)


class QuantinuumJob(QuantumJob):
    """Quantinuum job wrapper that handles the compile and execute phases.

    Quantinuum jobs go through two phases:
    1. Compilation: Circuits are compiled/optimized for the target device
    2. Execution: Compiled circuits are executed on the device

    To improve performance, this wrapper makes compilation non-blocking. When you submit
    a job, it returns immediately with a compile job. The execute job is created lazily
    once compilation completes.

    You can check the current phase using the `phase` property. The job will automatically
    transition from compile to execute phase when you call `status()` or `result()` after
    compilation completes.
    """

    def __init__(
        self,
        job_id: str,
        *,
        device: Any | None = None,
        _is_compile_job: bool = False,
        _shots: int | None = None,
        _backend_config: Any | None = None,
        _project: Any | None = None,
        _execute_name_prefix: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__(job_id, device)
        self._ref_obj = None
        self._cached_result = None
        # Compilation tracking
        self._is_compile_job = _is_compile_job
        self._shots = _shots
        self._backend_config = _backend_config
        self._project = _project
        self._execute_name_prefix = _execute_name_prefix
        self._execute_job_id = None

    @property
    def phase(self) -> str:
        """Return the current phase of the job: 'compile' or 'execute'.

        Returns:
            'compile' if the job is still in compilation phase
            'execute' if the job has transitioned to execution phase
        """
        return "compile" if self._is_compile_job else "execute"

    def _get_ref(self):
        # Lazily resolve and cache the qnexus job reference; fall back to id when unavailable
        if self._ref_obj is not None:
            return self._ref_obj
        try:
            self._ref_obj = qnx.jobs.get(self.id)
            return self._ref_obj
        except Exception:
            return self.id

    def _ensure_execute_job(self):
        """If this is a compile job, create the execute job once compilation is done."""
        if not self._is_compile_job:
            return  # Already an execute job

        if self._execute_job_id is not None:
            # Execute job already created, switch to tracking it
            self.id = self._execute_job_id
            self._is_compile_job = False
            self._ref_obj = None  # Clear cache to get new ref
            return

        # Check if compilation is complete
        compile_ref = self._get_ref()
        compile_status = getattr(compile_ref, "last_status", None)

        if compile_status != "COMPLETED":
            raise RuntimeError(
                f"Cannot get results - compilation not yet complete (status: {compile_status}). "
                f"Check job.status() or job.phase to monitor compilation progress."
            )

        # Get compiled circuits and create execute job
        from qnexus.models.language import Language

        compiled_refs = [item.get_output() for item in qnx.jobs.results(compile_ref)]
        execute_job = qnx.start_execute_job(
            programs=compiled_refs,
            name=self._execute_name_prefix,
            n_shots=[self._shots] * len(compiled_refs),
            backend_config=self._backend_config,
            project=self._project,
            language=Language.QIR,
        )

        self._execute_job_id = (
            getattr(execute_job, "id", None)
            or getattr(execute_job, "job_id", None)
            or str(execute_job)
        )

        # Switch to tracking the execute job
        self.id = self._execute_job_id
        self._is_compile_job = False
        self._ref_obj = None  # Clear cache

    def result(self) -> Result:
        # Return cached result if available
        if self._cached_result is not None:
            return self._cached_result

        # Ensure we're tracking the execute job, not compile job
        self._ensure_execute_job()

        ref = self._get_ref()
        results = qnx.jobs.results(ref)
        if not results:
            raise RuntimeError(f"No results available for job {self.id}")

        # Collect counts from all results (handles batch job submissions)
        # Always return a list for consistency, even with single circuit
        all_counts = []
        for result in results:
            counts = result.download_result().get_counts()
            norm_counts = {"".join(map(str, k)): v for k, v in counts.items()}
            all_counts.append(norm_counts)

        # Cache the result for future calls
        self._cached_result = Result(
            device_id=self._device.name if self._device else "unknown",
            job_id=self.id,
            success=True,
            data=GateModelResultData(measurement_counts=all_counts),
        )
        return self._cached_result

    def status(self) -> JobStatus:
        try:
            # If still a compile job, check compile status and maybe create execute job
            if self._is_compile_job:
                compile_ref = self._get_ref()
                compile_status = getattr(compile_ref, "last_status", None)

                if compile_status == "COMPLETED":
                    # Compilation done, create execute job
                    try:
                        self._ensure_execute_job()
                        # Now check execute job status
                        ref = self._get_ref()
                        last_status = getattr(ref, "last_status", None)
                    except Exception as e:
                        # Execute job creation failed, log and still consider as running
                        logger.warning(
                            f"Failed to create execute job for compile job {self.id}: {e}",
                            exc_info=True,
                        )
                        return JobStatus.RUNNING
                elif compile_status == "ERROR":
                    last_message = getattr(compile_ref, "last_message", None)
                    logger.error(f"Compilation job {self.id} failed: {last_message}")
                    return JobStatus.FAILED
                else:
                    # Compilation still in progress
                    return JobStatus.RUNNING
            else:
                # Already an execute job, check its status
                ref = self._get_ref()
                last_status = getattr(ref, "last_status", None)

            match last_status:
                case "COMPLETED":
                    return JobStatus.COMPLETED
                case "ERROR":
                    last_message = getattr(ref, "last_message", None)
                    logger.error(f"Execute job {self.id} failed: {last_message}")
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
