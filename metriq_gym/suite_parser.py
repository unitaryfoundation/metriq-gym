"""Parsing helpers for benchmark suite definitions."""

import json
from pathlib import Path
from typing import Any
from math import isclose

from pydantic import BaseModel, Field


class BenchmarkEntry(BaseModel):
    name: str = Field(..., description="Unique name for the benchmark in the suite")
    config: dict[str, Any] = Field(..., description="Benchmark configuration dictionary")
    weight: float = Field(
        1.0, ge=0, description="Relative weight of this benchmark in the suite score"
    )


class Suite(BaseModel):
    name: str = Field(..., description="Suite name")
    benchmarks: list[BenchmarkEntry] = Field(..., description="List of benchmarks in the suite")

    # Ensure suite weights sum to 1.0
    @classmethod
    def model_validate(cls, obj: Any):
        model = super().model_validate(obj)
        if model.benchmarks:
            total = sum(entry.weight for entry in model.benchmarks)
            if not isclose(total, 1.0):
                raise ValueError(
                    f"Invalid suite weights: sum is {total:.6g}, expected 1.0. "
                    "Set per-benchmark 'weight' so they sum to 1.0."
                )
        return model


def parse_suite_file(path: str | Path) -> Suite:
    """Parse a suite JSON file and return a Suite object."""
    with open(path, "r") as f:
        data = json.load(f)
    return Suite.model_validate(data)
