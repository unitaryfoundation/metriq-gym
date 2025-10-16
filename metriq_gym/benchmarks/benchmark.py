import argparse
from typing import Iterable, TYPE_CHECKING, Protocol

from pydantic import BaseModel
from dataclasses import dataclass

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


class SupportsId(Protocol):
    id: str


def flatten_job_ids(job: SupportsId | Iterable[SupportsId]) -> list[str]:
    """Return provider job IDs from a single job or an iterable of jobs."""
    if isinstance(job, Iterable) and not isinstance(job, (str, bytes)):
        return [job.id for job in job]
    return [job.id]


@dataclass
class BenchmarkData:
    """Stores intermediate data from pre-processing and dispatching"""

    provider_job_ids: list[str]

    @classmethod
    def from_quantum_job(cls, quantum_job, **kwargs):
        """Populate the provider job IDs from a QuantumJob or iterable of QuantumJobs."""
        return cls(provider_job_ids=flatten_job_ids(quantum_job), **kwargs)


class BenchmarkScore(BaseModel):
    value: float
    uncertainty: float = 0.0


class BenchmarkResult(BaseModel):
    """Base class for benchmark results with minimal author burden.

    Subclasses declare metric fields as numbers (float/int) or BenchmarkScore.
    - Numbers map to results.values[<field>] = number and results.uncertainties[...] = 0.0
    - BenchmarkScore maps to values[...] = value and uncertainties[...] = uncertainty
    """

    def _iter_metric_items(self):
        for name in self.__class__.model_fields:
            val = getattr(self, name, None)
            if val is None:
                continue
            if isinstance(val, BenchmarkScore):
                yield name, float(val.value), float(val.uncertainty)
            elif isinstance(val, (int, float)):
                yield name, float(val), 0.0
            # else: ignore non-numeric / non-score fields

    @property
    def values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v, _u in self._iter_metric_items():
            out[k] = v
        return out

    @property
    def uncertainties(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, _v, u in self._iter_metric_items():
            out[k] = u
        return out


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
