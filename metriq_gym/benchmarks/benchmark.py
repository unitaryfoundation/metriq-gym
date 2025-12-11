import argparse
from typing import Iterable, TYPE_CHECKING, Protocol
from abc import ABC

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

    def __str__(self) -> str:
        if self.uncertainty is None or self.uncertainty == "":
            return str(self.value)
        return f"{self.value} ± {self.uncertainty}"

    def __repr__(self) -> str:
        return f"BenchmarkScore(value={self.value}, uncertainty={self.uncertainty})"


class BenchmarkResult(BaseModel, ABC):
    """Base class for benchmark results.

    Subclasses declare metric fields as numbers (float/int) or BenchmarkScore.
    - Numbers map to results.values[<field>] = number and results.uncertainties[...] = None
    - BenchmarkScore maps to values[...] = value and uncertainties[...] = uncertainty (or None if unset)
    """

    def _iter_metric_items(self):
        for name in self.__class__.model_fields:
            if name in self.__class__.model_computed_fields:
                continue
            value = getattr(self, name, None)
            if isinstance(value, BenchmarkScore):
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

    def compute_score(self) -> BenchmarkScore | None:
        """Hook for computing a scalar score from result metrics.

        Default implementation returns None. Benchmarks should override this to
        implement single- or multi-metric scoring as appropriate.
        """
        return None

    @computed_field(return_type=BenchmarkScore | None)
    def score(self) -> BenchmarkScore | None:
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
