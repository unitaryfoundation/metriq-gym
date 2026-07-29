"""Parsing helpers for benchmark suite definitions."""

import json
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkEntry(BaseModel):
    name: str = Field(..., description="Unique name for the benchmark in the suite")
    component: str | None = Field(
        default=None,
        description=(
            "Optional benchmark-family identifier used to select related suite entries together. "
            "Defaults to the entry name for legacy suites."
        ),
    )
    config: dict[str, Any] = Field(..., description="Benchmark configuration dictionary")

    @property
    def component_name(self) -> str:
        """Return the component identifier used by suite selection."""
        return self.component or self.name


class Suite(BaseModel):
    name: str = Field(..., description="Suite name")
    version: str | None = Field(default=None, description="Version of the suite definition")
    description: str | None = Field(default=None, description="Description of the suite")
    source: str | None = Field(default=None, description="Source for the suite definition")
    full_suite_warning: str | None = Field(
        default=None,
        description=(
            "Warning shown before a full dispatch. Its presence requires an explicit --all opt-in."
        ),
    )
    benchmarks: list[BenchmarkEntry] = Field(..., description="List of benchmarks in the suite")

    @property
    def component_names(self) -> list[str]:
        """Return component identifiers in their first-occurrence order."""
        components: list[str] = []
        seen: set[str] = set()
        for benchmark in self.benchmarks:
            component_name = benchmark.component_name
            component_key = component_name.casefold()
            if component_key not in seen:
                seen.add(component_key)
                components.append(component_name)
        return components

    def select_components(self, requested_components: list[str]) -> list[BenchmarkEntry]:
        """Select all entries belonging to the requested components.

        Component matching is case-insensitive. The returned entries retain their
        order in the suite definition, regardless of option order.
        """
        requested_keys = [component.casefold() for component in requested_components]
        duplicate_keys = {
            component_key
            for component_key in requested_keys
            if requested_keys.count(component_key) > 1
        }
        if duplicate_keys:
            duplicates = sorted(
                {
                    component
                    for component in requested_components
                    if component.casefold() in duplicate_keys
                },
                key=str.casefold,
            )
            raise ValueError(f"Duplicate component selection: {', '.join(duplicates)}")

        available = {component.casefold(): component for component in self.component_names}
        unknown = [
            component for component in requested_components if component.casefold() not in available
        ]
        if unknown:
            raise ValueError(
                f"Unknown component(s): {', '.join(unknown)}. "
                f"Available components: {', '.join(self.component_names)}"
            )

        selected_keys = set(requested_keys)
        return [
            benchmark
            for benchmark in self.benchmarks
            if benchmark.component_name.casefold() in selected_keys
        ]


def parse_suite_file(path: str | Path) -> Suite:
    """Parse a suite JSON file or bundled suite name and return a Suite object."""
    candidate = Path(path)
    try:
        with candidate.open("r") as suite_file:
            data = json.load(suite_file)
    except FileNotFoundError:
        if candidate.parent != Path("."):
            raise

        resource_name = candidate.name if candidate.suffix else f"{candidate.name}.json"
        suite_resource = resources.files("metriq_gym").joinpath("suites", resource_name)
        if not suite_resource.is_file():
            raise
        with suite_resource.open("r") as suite_file:
            data = json.load(suite_file)

    return Suite.model_validate(data)
