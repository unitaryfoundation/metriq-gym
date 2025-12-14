from unittest.mock import MagicMock
from dataclasses import dataclass

import pytest
from pydantic import Field
from qbraid import QuantumJob
from metriq_gym.benchmarks.benchmark import (
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)


class TestBenchmarkData:
    def test_from_quantum_job_single(self):
        TEST_JOB_ID = "test_job_id"
        mock_job = MagicMock(spec=QuantumJob)
        mock_job.id = TEST_JOB_ID
        data = BenchmarkData.from_quantum_job(mock_job)
        assert data.provider_job_ids == [TEST_JOB_ID]

    def test_from_quantum_job_iterable(self):
        TEST_JOB_IDS = ["job1", "job2"]
        mock_job1 = MagicMock(spec=QuantumJob)
        mock_job1.id = TEST_JOB_IDS[0]
        mock_job2 = MagicMock(spec=QuantumJob)
        mock_job2.id = TEST_JOB_IDS[1]
        jobs = [mock_job1, mock_job2]
        data = BenchmarkData.from_quantum_job(jobs)
        assert data.provider_job_ids == TEST_JOB_IDS

    def test_from_quantum_job_with_kwargs(self):
        @dataclass
        class CustomBenchmarkData(BenchmarkData):
            extra: int = 0

        TEST_JOB_ID = "job_with_extra"
        mock_job = MagicMock(spec=QuantumJob)
        mock_job.id = TEST_JOB_ID
        data = CustomBenchmarkData.from_quantum_job(mock_job, extra=42)
        assert data.provider_job_ids == [TEST_JOB_ID]
        assert data.extra == 42


class TestBenchmarkDirections:
    def test_no_direction_required_for_float_metric(self):
        class R(BenchmarkResult):
            numeric_value: float

            def compute_score(self):
                return BenchmarkScore(value=self.numeric_value, uncertainty=None)

        # Should not raise
        r = R(numeric_value=1.23)
        assert r.values["numeric_value"] == pytest.approx(1.23)

    def test_benchmarkscore_metric_allowed(self):
        class R(BenchmarkResult):
            metric: BenchmarkScore

            def compute_score(self):
                return self.metric

        r = R(metric=BenchmarkScore(value=0.5, uncertainty=0.1))
        assert r.values["metric"] == pytest.approx(0.5)
        assert r.uncertainties["metric"] == pytest.approx(0.1)

    def test_bool_metric_allowed(self):
        class R(BenchmarkResult):
            ok: bool

            def compute_score(self):
                return float(self.ok)

        r = R(ok=True)
        assert r.values["ok"] == 1.0

    def test_fields_with_metadata_are_ok(self):
        class R(BenchmarkResult):
            accuracy: float = Field(...)
            latency: float = Field(...)

            def compute_score(self):
                return BenchmarkScore(value=self.accuracy, uncertainty=None)

        r = R(accuracy=0.99, latency=12.0)
        assert r.values["accuracy"] == pytest.approx(0.99)
        assert "latency" in r.values
