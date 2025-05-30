import json
import logging
from unittest.mock import patch
import pytest
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


def test_load_jobs_empty_file(job_manager):
    assert job_manager.get_jobs() == []


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


def test_load_jobs_with_invalid_entries(tmpdir, caplog):
    """Test that JobManager gracefully handles invalid job entries in the JSONL file."""
    jobs_file = tmpdir.join("test_jobs_with_invalid.jsonl")
    JobManager.jobs_file = str(jobs_file)
    
    # Create a valid job
    valid_job = MetriqGymJob(
        id="valid_job_id",
        provider_name="test_provider",
        device_name="test_device",
        job_type=FakeJobType(FAKE_BENCHMARK_NAME),
        params={},
        data={},
        dispatch_time=datetime.now(),
    )
    
    # Write a mix of valid and invalid job entries to the file
    with open(jobs_file, "w") as f:
        # Valid job
        f.write(valid_job.serialize() + "\n")
        
        # Invalid JSON (malformed)
        f.write('{"invalid": "json", "missing_fields": true\n')
        
        # Valid JSON but invalid job type
        invalid_job_data = {
            "id": "invalid_job_type_id",
            "job_type": "NONEXISTENT_TYPE",
            "params": {},
            "data": {},
            "provider_name": "test_provider",
            "device_name": "test_device",
            "dispatch_time": "2023-01-01T00:00:00"
        }
        f.write(json.dumps(invalid_job_data) + "\n")
        
        # Valid JSON but invalid datetime format
        invalid_datetime_job = {
            "id": "invalid_datetime_id",
            "job_type": FAKE_BENCHMARK_NAME,
            "params": {},
            "data": {},
            "provider_name": "test_provider",
            "device_name": "test_device",
            "dispatch_time": "invalid-datetime-format"
        }
        f.write(json.dumps(invalid_datetime_job) + "\n")
        
        # Another valid job
        valid_job2 = MetriqGymJob(
            id="valid_job_id_2",
            provider_name="test_provider_2",
            device_name="test_device_2",
            job_type=FakeJobType(FAKE_BENCHMARK_NAME),
            params={},
            data={},
            dispatch_time=datetime.now(),
        )
        f.write(valid_job2.serialize() + "\n")
    
    # Set up logging capture
    caplog.set_level(logging.WARNING)
    
    # Load jobs and verify behavior
    job_manager = JobManager()
    jobs = job_manager.get_jobs()
    
    # Should load only the 2 valid jobs
    assert len(jobs) == 2
    assert jobs[0].id == "valid_job_id"
    assert jobs[1].id == "valid_job_id_2"
    
    # Should have logged warnings for the 3 invalid entries
    warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warning_messages) == 3
    
    # Check that warning messages contain expected information
    assert any("line 2" in msg for msg in warning_messages)
    assert any("line 3" in msg for msg in warning_messages) 
    assert any("line 4" in msg for msg in warning_messages)
    assert all(str(jobs_file) in msg for msg in warning_messages)