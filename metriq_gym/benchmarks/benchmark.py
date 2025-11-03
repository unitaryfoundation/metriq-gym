import argparse
from typing import Iterable, TYPE_CHECKING, Protocol
from enum import StrEnum

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


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


class BenchmarkScore(BaseModel):
    value: float
    # If not specified, treat uncertainty as not available (N/A)
    uncertainty: float | None = None


class BenchmarkResult(BaseModel):
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

    @property
    def directions(self) -> dict[str, str]:
        d: dict[str, str] = {}
        for name, field in self.__class__.model_fields.items():
            value = getattr(self, name, None)
            # Only include metrics which are simple numbers or BenchmarkScore
            if isinstance(value, (BenchmarkScore, float)) or type(value) is int:
                extra = getattr(field, "json_schema_extra", None) or {}
                direction = extra.get("direction", MetricDirection.HIGHER.value)
                if isinstance(direction, MetricDirection):
                    direction = direction.value
                direction = str(direction).lower()
                d[name] = (
                    direction if direction in ("higher", "lower") else MetricDirection.HIGHER.value
                )
        return d

    @model_validator(mode="after")
    def _validate_metric_directions(self) -> "BenchmarkResult":
        missing: list[str] = []
        for name, field in self.__class__.model_fields.items():
            value = getattr(self, name, None)
            # Enforce direction only for primary numeric metrics: float or BenchmarkScore
            if isinstance(value, BenchmarkScore) or isinstance(value, float):
                extra = getattr(field, "json_schema_extra", None) or {}
                if "direction" not in (extra or {}):
                    missing.append(name)
        if missing:
            raise ValueError(
                "Missing metric direction for: "
                + ", ".join(missing)
                + '. Define Field(..., json_schema_extra={"direction": MetricDirection.HIGHER|LOWER}).'
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
