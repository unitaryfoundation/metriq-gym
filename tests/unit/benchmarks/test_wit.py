from datetime import datetime

import pytest

from metriq_gym.benchmarks.wit import WITResult
from pydantic import Field
from metriq_gym.benchmarks.benchmark import BenchmarkScore, BenchmarkResult
from metriq_gym.constants import JobType
from metriq_gym.helpers.statistics import (
    binary_expectation_stddev,
    binary_expectation_value,
)
from metriq_gym.exporters.base_exporter import BaseExporter
from metriq_gym.job_manager import MetriqGymJob


class _DummyExporter(BaseExporter):
    def export(self) -> None:  # pragma: no cover - not used in tests
        raise NotImplementedError


def _build_metriq_job() -> MetriqGymJob:
    return MetriqGymJob(
        id="test-job",
        job_type=JobType.WIT,
        params={"benchmark_name": "WIT", "shots": 10},
        data={"provider_job_ids": ["qid"]},
        provider_name="provider",
        device_name="device",
        dispatch_time=datetime.now(),
    )


def test_calculate_expectation_value_uses_effective_shots():
    counts = {"1": 6, "0": 4}
    assert binary_expectation_value(10, counts) == pytest.approx(0.6)

    counts_truncated = {"1": 3}
    # Only three total shots should be used to avoid inflating the expectation value.
    assert binary_expectation_value(10, counts_truncated) == pytest.approx(1.0)


def test_calculate_expectation_value_handles_zero_counts():
    assert binary_expectation_value(100, {}) == 0.0


def test_calculate_expectation_value_error_binomial_uncertainty():
    counts = {"1": 60, "0": 40}
    err = binary_expectation_stddev(100, counts)
    assert err == pytest.approx(0.049, abs=1e-3)


def test_calculate_expectation_value_error_handles_zero_counts():
    assert binary_expectation_stddev(100, {}) == 0.0


def test_wit_result_exports_symmetric_results_and_uncertainties():
    job = _build_metriq_job()
    result = WITResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))
    exporter = _DummyExporter(job, result)

    payload = exporter.as_dict()
    assert payload["results"]["values"] == {"expectation_value": pytest.approx(0.5)}
    assert payload["results"]["uncertainties"] == {"expectation_value": pytest.approx(0.05)}
    assert payload["platform"] == {"provider": "provider", "device": "device"}
    assert result.values == pytest.approx({"expectation_value": 0.5})
    assert result.uncertainties == pytest.approx({"expectation_value": 0.05})


def test_wit_result_includes_score_in_export():
    job = _build_metriq_job()
    result = WITResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))
    exporter = _DummyExporter(job, result)

    payload = exporter.as_dict()
    assert payload["results"]["score"] == pytest.approx(0.5)


def test_wit_result_score_properties():
    r = WITResult(expectation_value=BenchmarkScore(value=0.33, uncertainty=0.01))
    assert r.score == pytest.approx(0.33)


def test_missing_direction_no_longer_raises_validation_error():
    class DummyResult(BenchmarkResult):
        metric: BenchmarkScore

        def compute_score(self):
            return None

    r = DummyResult(metric=BenchmarkScore(value=1.0, uncertainty=0.0))
    assert r.values["metric"] == pytest.approx(1.0)


def test_metadata_is_optional_and_ignored_for_scoring():
    class DummyResult2(BenchmarkResult):
        latency: BenchmarkScore = Field(...)

        def compute_score(self):
            return None

    r = DummyResult2(latency=BenchmarkScore(value=12.3, uncertainty=0.5))
    assert r.values["latency"] == pytest.approx(12.3)


def test_wit_result_uncertainty_keys_match_values():
    r = WITResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))
    assert set(r.values.keys()) == {"expectation_value"}
    assert set(r.uncertainties.keys()) == {"expectation_value"}
