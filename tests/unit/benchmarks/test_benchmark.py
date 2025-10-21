from unittest.mock import MagicMock
from dataclasses import dataclass

import pytest
from pydantic import Field
from qbraid import QuantumJob
from metriq_gym.benchmarks.benchmark import (
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
    MetricDirection,
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
    def test_direction_required_for_float_metric(self):
        class R(BenchmarkResult):
            score: float  # missing direction

        with pytest.raises(ValueError):
            R(score=1.23)

    def test_direction_required_for_benchmarkscore_metric(self):
        class R(BenchmarkResult):
            metric: BenchmarkScore  # missing direction

        with pytest.raises(ValueError):
            R(metric=BenchmarkScore(value=0.5, uncertainty=0.1))

    def test_direction_not_required_for_bool(self):
        class R(BenchmarkResult):
            ok: bool

        r = R(ok=True)
        assert r.directions == {}

    def test_directions_from_field_metadata_float(self):
        class R(BenchmarkResult):
            accuracy: float = Field(..., json_schema_extra={"direction": MetricDirection.HIGHER})

        r = R(accuracy=0.99)
        assert r.directions == {"accuracy": "higher"}

    def test_directions_from_field_metadata_benchmarkscore(self):
        class R(BenchmarkResult):
            latency: BenchmarkScore = Field(
                ..., json_schema_extra={"direction": MetricDirection.LOWER}
            )

        r = R(latency=BenchmarkScore(value=12.0, uncertainty=0.5))
        assert r.directions == {"latency": "lower"}
