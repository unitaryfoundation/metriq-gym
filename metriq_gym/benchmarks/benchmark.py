import argparse
from typing import Iterable, TYPE_CHECKING, Any

from pydantic import BaseModel
from dataclasses import dataclass

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


def flatten_job_ids(quantum_job: Any | Iterable[Any]) -> list[str]:
    """Return provider job IDs from a single job or an iterable of jobs.

    Uses duck-typing to avoid importing heavy qbraid types at runtime.
    """
    try:
        if hasattr(quantum_job, "id") and not isinstance(quantum_job, (str, bytes)):
            return [quantum_job.id]
    except Exception:
        pass

    if isinstance(quantum_job, Iterable) and not isinstance(quantum_job, (str, bytes, dict)):
        return [job.id for job in quantum_job]

    raise TypeError(f"Unsupported job type: {type(quantum_job)}")


@dataclass
class BenchmarkData:
    """Stores intermediate data from pre-processing and dispatching"""

    provider_job_ids: list[str]

    @classmethod
    def from_quantum_job(cls, quantum_job, **kwargs):
        """Populate the provider job IDs from a QuantumJob or iterable of QuantumJobs."""
        return cls(provider_job_ids=flatten_job_ids(quantum_job), **kwargs)


class BenchmarkResult(BaseModel):
    """Stores the final results of the benchmark"""

    pass


class Benchmark[BD: BenchmarkData, BR: BenchmarkResult]:
    def __init__(
        self,
        args: argparse.Namespace,
        params: BaseModel,
    ):
        self.args = args
        self.params: BaseModel = params

    def dispatch_handler(self, device: "QuantumDevice") -> BD:
        raise NotImplementedError

    def poll_handler(
        self,
        job_data: BD,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> BR:
        raise NotImplementedError
