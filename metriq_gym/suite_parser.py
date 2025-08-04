import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class BenchmarkEntry(BaseModel):
    name: str = Field(..., description="Unique name for the benchmark in the suite")
    config: Dict[str, Any] = Field(..., description="Benchmark configuration dictionary")


class Suite(BaseModel):
    name: str = Field(..., description="Suite name")
    benchmarks: List[BenchmarkEntry] = Field(..., description="List of benchmarks in the suite")


def parse_suite_file(path: str | Path) -> Suite:
    """Parse a suite JSON file and return a Suite object."""
    with open(path, "r") as f:
        data = json.load(f)
    return Suite.model_validate(data)
