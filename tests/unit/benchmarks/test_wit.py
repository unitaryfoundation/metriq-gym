from datetime import datetime

import pytest

from metriq_gym.benchmarks.wit import WITResult
from metriq_gym.benchmarks.benchmark import BenchmarkScore
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


def test_wit_result_uncertainty_keys_match_values():
    r = WITResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))
    assert set(r.values.keys()) == {"expectation_value"}
    assert set(r.uncertainties.keys()) == {"expectation_value"}
