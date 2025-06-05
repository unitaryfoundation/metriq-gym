from unittest.mock import patch
import pytest
import logging
from datetime import datetime
from metriq_gym.job_manager import JobManager, MetriqGymJob
from tests.test_schema_validator import FAKE_BENCHMARK_NAME, FakeJobType


@pytest.fixture(autouse=True)
def patch_job_type_enum():
    with patch("metriq_gym.job_manager.JobType", FakeJobType):
        yield


@pytest.fixture
def job_manager(tmpdir):
    jobs_file = tmpdir.join("test_jobs.jsonl")
    JobManager.jobs_file = str(jobs_file)
    return JobManager()


@pytest.fixture
def sample_job():
    return MetriqGymJob(
        id="test_job_id",
        provider_name="test_provider",
        device_name="test_device",
        job_type=FakeJobType(FAKE_BENCHMARK_NAME),
        params={},
        data={},
        dispatch_time=datetime.now(),
    )


def test_load_jobs_empty_file(tmp_path, caplog):
    """Test loading jobs from an empty file."""
    # Create empty file at the hardcoded path
    jobs_file = tmp_path / ".metriq_gym_jobs.jsonl"
    jobs_file.write_text("")
    
    # Temporarily modify the class variable
    original_jobs_file = JobManager.jobs_file
    JobManager.jobs_file = str(jobs_file)
    
    try:
        # Set up log capture before creating JobManager
        with caplog.at_level(logging.WARNING):
            manager = JobManager()
            assert len(manager.jobs) == 0
            assert "No valid jobs found" in caplog.text
    finally:
        # Restore original jobs_file path
        JobManager.jobs_file = original_jobs_file


def test_add_job(job_manager, sample_job):
    job_manager.add_job(sample_job)
    jobs = job_manager.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == sample_job.id


def test_load_jobs_with_existing_data(job_manager, sample_job):
    job_manager.add_job(sample_job)
    new_job_manager = JobManager()
    jobs = new_job_manager.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == sample_job.id


def test_load_jobs_skips_invalid(tmp_path, caplog):
    """Test that invalid jobs are skipped with appropriate warnings."""
    # Create file with mix of valid and invalid jobs at the hardcoded path
    jobs_file = tmp_path / ".metriq_gym_jobs.jsonl"
    jobs_file.write_text(f"""{{"id": "valid1", "job_type": "{FAKE_BENCHMARK_NAME}", "params": {{}}, "data": {{"provider_job_ids": []}}, "provider_name": "test", "device_name": "test", "dispatch_time": "2024-01-01T00:00:00"}}
{{"id": "invalid1"}}
{{"invalid": "json"}}
{{"id": "valid2", "job_type": "{FAKE_BENCHMARK_NAME}", "params": {{}}, "data": {{"provider_job_ids": []}}, "provider_name": "test", "device_name": "test", "dispatch_time": "2024-01-01T00:00:00"}}""")
    
    # Temporarily modify the class variable
    original_jobs_file = JobManager.jobs_file
    JobManager.jobs_file = str(jobs_file)
    
    try:
        with caplog.at_level(logging.WARNING):
            manager = JobManager()
            
            # Should only load the two valid jobs
            assert len(manager.jobs) == 2
            assert any(job.id == "valid1" for job in manager.jobs)
            assert any(job.id == "valid2" for job in manager.jobs)
            
            # Check that appropriate warnings were logged
            log_text = caplog.text
            assert "Line 2: MetriqGymJob.__init__() missing 6 required positional arguments" in log_text
            assert "Line 3: MetriqGymJob.__init__() got an unexpected keyword argument 'invalid'" in log_text
    finally:
        # Restore original jobs_file path
        JobManager.jobs_file = original_jobs_file


def test_load_jobs_whitespace_only(job_manager, caplog):
    """Test handling of file with only whitespace."""
    with open(JobManager.jobs_file, "w") as f:
        f.write("\n\n  \n\t\n")
    
    with caplog.at_level(logging.WARNING):
        new_job_manager = JobManager()
        assert new_job_manager.get_jobs() == []
        assert "No valid jobs found" in caplog.text
