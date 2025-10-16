import argparse
from typing import Any, Iterable, TYPE_CHECKING, Protocol

from pydantic import BaseModel, model_validator
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


class BenchmarkResult(BaseModel):
    """Stores the final results of the benchmark."""

    values: dict[str, Any] | None = None
    uncertainties: dict[str, Any] | None = None

    def result_metrics(self) -> dict[str, Any]:
        """Return benchmark metrics to expose under the results dictionary."""
        if self.values:
            return self.values
        base = self.model_dump(exclude={"values", "uncertainties"}, exclude_none=True)
        return base

    def uncertainty_metrics(self) -> dict[str, Any]:
        """Return statistical or systematic uncertainties for the benchmark metrics."""
        return self.uncertainties or {}

    @model_validator(mode="after")
    def _validate_uncertainty_keys(self) -> "BenchmarkResult":
        """Ensure any exposed uncertainty metrics align with the reported results."""
        result_keys = set(self.result_metrics().keys())
        uncertainty_keys = set(self.uncertainty_metrics().keys())
        missing_values = uncertainty_keys - result_keys
        if missing_values:
            raise ValueError(
                "Uncertainty keys must correspond to existing results. Missing values for: "
                f"{missing_values}"
            )
        return self


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
