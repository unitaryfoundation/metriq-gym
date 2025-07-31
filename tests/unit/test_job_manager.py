from unittest.mock import patch
import pytest
import json
import logging
from datetime import datetime
from metriq_gym.job_manager import JobManager, MetriqGymJob
from tests.unit.test_schema_validator import FAKE_BENCHMARK_NAME, FakeJobType


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


def test_mixed_valid_and_invalid_jobs_are_handled_gracefully(tmpdir, caplog):
    """
    Test that JobManager loads only valid jobs from a file containing a mix
    of valid and invalid entries, and logs appropriate warnings for each invalid entry.
    """
    jobs_file = tmpdir.join("mixed_jobs.jsonl")
    JobManager.jobs_file = str(jobs_file)

    # Prepare job entries
    valid_job_1 = MetriqGymJob(
        id="valid_job_1",
        provider_name="provider_1",
        device_name="device_1",
        job_type=FakeJobType(FAKE_BENCHMARK_NAME),
        params={},
        data={"provider_job_ids": []},
        dispatch_time=datetime.now(),
    )
    valid_job_2 = MetriqGymJob(
        id="valid_job_2",
        provider_name="provider_2",
        device_name="device_2",
        job_type=FakeJobType(FAKE_BENCHMARK_NAME),
        params={},
        data={"provider_job_ids": []},
        dispatch_time=datetime.now(),
    )

    with open(jobs_file, "w") as f:
        f.write(valid_job_1.serialize() + "\n")  # Valid
        f.write("\n")  # Empty line
        f.write('{"invalid": "json", "missing": true\n')  # Malformed JSON
        f.write(
            json.dumps(
                {  # Missing fields
                    "id": "missing_fields",
                    "params": {},
                    "data": {},
                    "provider_name": "test",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {  # Invalid job type
                    "id": "bad_job_type",
                    "job_type": "NONEXISTENT_TYPE",
                    "params": {},
                    "data": {},
                    "provider_name": "test",
                    "device_name": "device",
                    "dispatch_time": "2024-01-01T00:00:00",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {  # Bad datetime
                    "id": "bad_datetime",
                    "job_type": FAKE_BENCHMARK_NAME,
                    "params": {},
                    "data": {},
                    "provider_name": "test",
                    "device_name": "device",
                    "dispatch_time": "not-a-datetime",
                }
            )
            + "\n"
        )
        f.write(valid_job_2.serialize() + "\n")  # Valid

    caplog.set_level(logging.WARNING)
    manager = JobManager()
    loaded_jobs = manager.get_jobs()

    # Expect only 2 valid jobs to be loaded
    assert len(loaded_jobs) == 2
    assert loaded_jobs[0].id == "valid_job_1"
    assert loaded_jobs[1].id == "valid_job_2"

    # Expect 4 warning logs for invalid entries (excluding blank line)
    warnings = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 4

    # Verify log contents by error type
    joined_logs = " ".join(warnings)
    assert "Invalid JSON at position" in joined_logs
    assert (
        "Missing required field" in joined_logs
        or "Incorrect data structure" in joined_logs
        or "Data structure mismatch" in joined_logs
    )
    assert "Unknown job type:" in joined_logs
    assert "Invalid datetime format:" in joined_logs or "Bad datetime format:" in joined_logs


def test_load_jobs_with_only_invalid_entries(tmpdir, caplog):
    """Test that a warning is logged when no valid jobs are found."""
    jobs_file = tmpdir.join("test_jobs_only_invalid.jsonl")
    JobManager.jobs_file = str(jobs_file)

    with open(jobs_file, "w") as f:
        f.write('{"bad": "entry", "no_required_fields": true}\n')  # Invalid entry

    caplog.set_level(logging.WARNING)
    job_manager = JobManager()
    jobs = job_manager.get_jobs()

    assert jobs == []  # No valid jobs
    warning_messages = [record.message for record in caplog.records]
    assert any("No valid jobs found" in msg for msg in warning_messages)


def test_load_jobs_file_missing(tmpdir):
    """Test that no errors occur and jobs list is empty when jobs file is missing."""
    missing_file_path = tmpdir.join("non_existent.jsonl")
    JobManager.jobs_file = str(missing_file_path)

    job_manager = JobManager()
    assert job_manager.get_jobs() == []


def test_delete_job(job_manager, sample_job):
    job_manager.add_job(sample_job)
    job_manager.delete_job(sample_job.id)
    jobs = job_manager.get_jobs()
    assert len(jobs) == 0


def test_job_version_serialization_and_export(monkeypatch):
    """Ensure stored version persists across serialization and is used by exporters."""
    import importlib.metadata
    from metriq_gym.exporters.json_exporter import JsonExporter
    from metriq_gym.benchmarks.benchmark import BenchmarkResult

    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.0")
    job = MetriqGymJob(
        id="ver_job",
        provider_name="provider",
        device_name="device",
        job_type=FakeJobType(FAKE_BENCHMARK_NAME),
        params={},
        data={},
        dispatch_time=datetime.now(),
    )

    assert job.version == "1.0"
    serialized = job.serialize()

    monkeypatch.setattr(importlib.metadata, "version", lambda _: "2.0")
    loaded_job = MetriqGymJob.deserialize(serialized)
    assert loaded_job.version == "1.0"

    exporter = JsonExporter(loaded_job, BenchmarkResult())
    export_dict = exporter.as_dict()
    assert export_dict["version"] == "1.0"
