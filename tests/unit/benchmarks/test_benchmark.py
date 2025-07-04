from unittest.mock import MagicMock
from dataclasses import dataclass

from qbraid import QuantumJob
from metriq_gym.benchmarks.benchmark import BenchmarkData


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
