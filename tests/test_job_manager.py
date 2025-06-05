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


def test_load_jobs_empty_file(job_manager, caplog):
    """Test handling of empty file."""
    caplog.set_level(logging.WARNING)
    assert job_manager.get_jobs() == []
    assert "No valid jobs found" in caplog.text


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


def test_load_jobs_skips_invalid(job_manager, sample_job, caplog):
    """Test that invalid JSONL entries are skipped with appropriate warnings."""
    # Add a valid job
    job_manager.add_job(sample_job)
    
    # Add various invalid entries
    with open(JobManager.jobs_file, "a") as f:
        f.write('{"id": "invalid"}\n')  # Missing required fields
        f.write('{"invalid": "json"}\n')  # Invalid JSON
        f.write('{"id": "test", "job_type": "invalid", "params": {}, "device_name": "test"}\n')  # Invalid job_type
        f.write('{"id": "test", "job_type": "BSEQ", "params": "not_dict", "device_name": "test"}\n')  # Invalid params
        f.write('{"id": "test", "job_type": "BSEQ", "params": {}, "device_name": 123}\n')  # Invalid device_name
    
    caplog.set_level(logging.WARNING)
    new_job_manager = JobManager()
    
    # Verify only valid job was loaded
    jobs = new_job_manager.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == sample_job.id
    
    # Verify warnings were logged
    log_text = caplog.text
    assert "Invalid JSON" in log_text
    assert "Invalid or missing job_type" in log_text
    assert "Invalid or missing params" in log_text
    assert "Invalid or missing device_name" in log_text


def test_load_jobs_whitespace_only(job_manager, caplog):
    """Test handling of file with only whitespace."""
    with open(JobManager.jobs_file, "w") as f:
        f.write("\n\n  \n\t\n")
    
    caplog.set_level(logging.WARNING)
    new_job_manager = JobManager()
    assert new_job_manager.get_jobs() == []
    assert "No valid jobs found" in caplog.text
