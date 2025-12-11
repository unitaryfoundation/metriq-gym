import pytest
from pydantic import Field

from metriq_gym.benchmarks.benchmark import (
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.exporters.base_exporter import BaseExporter


class _DummyExporter(BaseExporter):
    def export(self) -> None:  # pragma: no cover - not used in tests
        raise NotImplementedError


def test_payload_includes_null_uncertainty_for_numeric_and_benchmarkscore(metriq_job):
    # Numeric metric should have uncertainty None (-> null in JSON)
    class _NumResult(BenchmarkResult):
        numeric_metric: float = Field(...)

    num_result = _NumResult(numeric_metric=1.23)
    num_payload = _DummyExporter(metriq_job, num_result).as_dict()

    assert num_payload["results"]["numeric_metric"] == pytest.approx(1.23)
    assert "score" not in num_payload["results"]

    # BenchmarkScore without uncertainty should also be None
    class _WitLikeResult(BenchmarkResult):
        expectation_value: BenchmarkScore = Field(...)

    bs_result = _WitLikeResult(expectation_value=BenchmarkScore(value=0.7))
    bs_payload = _DummyExporter(metriq_job, bs_result).as_dict()

    assert bs_payload["results"]["expectation_value"]["value"] == pytest.approx(0.7)
    assert bs_payload["results"]["expectation_value"]["uncertainty"] is None
