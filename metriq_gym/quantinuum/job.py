from typing import Any
from uuid import UUID

import qnexus as qnx
from qbraid.runtime import GateModelResultData, JobStatus, QuantumJob, Result


class QuantinuumJob(QuantumJob):
    def __init__(self, job_id: str, *, device: Any | None = None, **_: Any) -> None:
        super().__init__(job_id, device)

    def _ref(self):
        try:
            # Prefer concrete job reference if supported by SDK
            return qnx.jobs.get(self.id)
        except Exception:
            try:
                return qnx.jobs.get(UUID(str(self.id)))
            except Exception:
                return self.id

    def result(self) -> Result:
        ref = self._ref()
        results = qnx.jobs.results(ref)
        if not results:
            qnx.jobs.wait_for(ref)
            results = qnx.jobs.results(ref)
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

        # Reduce any key to a single bit '0'/'1' expected by Wormhole
        def _last_bit(k: Any) -> str:
            # tuple/list: take last element recursively
            while isinstance(k, (tuple, list)) and len(k) > 0:
                k = k[-1]
            # string of bits: take last character
            if isinstance(k, str):
                s = k.strip()
                if s and all(ch in "01" for ch in s):
                    return s[-1]
                return "1" if s in {"1", "True", "true"} else "0"
            # int/bool
            if isinstance(k, (int, bool)):
                return "1" if (int(k) == 1) else "0"
            # fallback
            try:
                return "1" if int(k) == 1 else "0"
            except Exception:
                return "0"

        norm_counts: dict[str, int] = {"0": 0, "1": 0}
        for k, v in counts.items():
            b = _last_bit(k)
            norm_counts[b] += int(v)

        return Result(
            device_id="quantinuum",
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
