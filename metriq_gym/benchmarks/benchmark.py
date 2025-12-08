import argparse
from typing import Iterable, TYPE_CHECKING, Protocol
from abc import ABC, abstractmethod

from pydantic import BaseModel, computed_field
from dataclasses import dataclass


if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob
    from metriq_gym.resource_estimation import CircuitBatch


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
    # If not specified, treat uncertainty as not available (N/A)
    uncertainty: float | None = None


class BenchmarkResult(BaseModel, ABC):
    """Base class for benchmark results.

    Subclasses declare metric fields as numbers (float/int) or BenchmarkScore.
    - Numbers map to results.values[<field>] = number and results.uncertainties[...] = None
    - BenchmarkScore maps to values[...] = value and uncertainties[...] = uncertainty (or None if unset)
    """

    def _iter_metric_items(self):
        for name in self.__class__.model_fields:
            value = getattr(self, name, None)
            if isinstance(value, BenchmarkScore):
                # If uncertainty is not provided, leave as None
                u = value.uncertainty
                yield name, float(value.value), (float(u) if u is not None else None)
            elif isinstance(value, (int, float)):
                # Bare numeric metrics have unspecified uncertainty
                yield name, float(value), None

    @property
    def values(self) -> dict[str, float]:
        return {name: value for name, value, _ in self._iter_metric_items()}

    @property
    def uncertainties(self) -> dict[str, float | None]:
        return {name: uncertainty for name, _, uncertainty in self._iter_metric_items()}

    @abstractmethod
    def compute_score(self) -> float | None:
        """Hook for computing a scalar score from result metrics.

        Default implementation returns None. Benchmarks should override this to
        implement single- or multi-metric scoring as appropriate.
        """
        ...

    @computed_field(return_type=float | None)
    def score(self) -> float | None:
        return self.compute_score()


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

    def estimate_resources_handler(self, device: "QuantumDevice") -> list["CircuitBatch"]:
        raise NotImplementedError
