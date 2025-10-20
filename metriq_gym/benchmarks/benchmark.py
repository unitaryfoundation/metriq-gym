import argparse
from typing import Iterable, TYPE_CHECKING, Protocol
from enum import StrEnum

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


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


class BenchmarkScore(BaseModel):
    value: float
    uncertainty: float = 0.0
    direction: MetricDirection = MetricDirection.HIGHER


class BenchmarkResult(BaseModel):
    """Base class for benchmark results.

    Subclasses declare metric fields as numbers (float/int) or BenchmarkScore.
    - Numbers map to results.values[<field>] = number and results.uncertainties[...] = 0.0
    - BenchmarkScore maps to values[...] = value and uncertainties[...] = uncertainty
    """

    def _iter_metric_items(self):
        for name in self.__class__.model_fields:
            value = getattr(self, name, None)
            if isinstance(value, BenchmarkScore):
                yield name, float(value.value), float(value.uncertainty)
            elif isinstance(value, (int, float)):
                yield name, float(value), 0.0

    @property
    def values(self) -> dict[str, float]:
        return {name: value for name, value, _ in self._iter_metric_items()}

    @property
    def uncertainties(self) -> dict[str, float]:
        return {name: uncertainty for name, _, uncertainty in self._iter_metric_items()}

    @property
    def directions(self) -> dict[str, str]:
        d: dict[str, str] = {}
        for name in self.__class__.model_fields:
            value = getattr(self, name, None)
            if isinstance(value, BenchmarkScore):
                d[name] = value.direction.value
            elif isinstance(value, (int, float)):
                d[name] = MetricDirection.HIGHER.value
        return d


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
