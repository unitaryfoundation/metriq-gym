import argparse
from collections.abc import Iterable

from pydantic import BaseModel
from dataclasses import dataclass

from qbraid import GateModelResultData, QuantumDevice, QuantumJob


def flatten_job_ids(quantum_job: QuantumJob | Iterable[QuantumJob]) -> list[str]:
    if isinstance(quantum_job, QuantumJob):
        return [quantum_job.id]
    elif isinstance(quantum_job, Iterable):
        return [job.id for job in quantum_job]
    else:
        raise TypeError(f"Unsupported job type: {type(quantum_job)}")


@dataclass
class BenchmarkData:
    """Stores intermediate data from pre-processing and dispatching"""

    provider_job_ids: list[str]

    @classmethod
    def from_quantum_job(cls, quantum_job, **kwargs):
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

    def dispatch_handler(self, device: QuantumDevice) -> BD:
        raise NotImplementedError

    def poll_handler(
        self,
        job_data: BD,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> BR:
        raise NotImplementedError
