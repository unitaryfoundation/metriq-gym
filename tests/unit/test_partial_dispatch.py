"""Offline regressions for interrupted acquisitions and durable submission IDs."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qbraid.runtime import JobStatus

from metriq_gym.benchmarks.benchmark import BenchmarkData
from metriq_gym.constants import JobType
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.run import dispatch_job, dispatch_suite, fetch_result


def _dispatch(tmp_path, monkeypatch, *, suite=False, failure=None):
    from metriq_gym.qplatform.submission import run_circuits

    config = {"benchmark_name": "WIT", "num_qubits": 7, "shots": 10}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(
        json.dumps({"name": "test", "benchmarks": [{"name": "wit", "config": config}]})
    )
    args = SimpleNamespace(
        config=str(config_file), suite_config=str(suite_file), provider="local", device="fake"
    )
    manager = JobManager(tmp_path / "jobs.jsonl")
    device = Mock()
    device.id = "fake"
    device.num_qubits = 7
    observations = []

    def submit(*args, **kwargs):
        # Inspect disk, rather than the mutable object held by the dispatcher.
        persisted = JobManager(manager.jobs_file).get_jobs()
        observations.append(persisted[0].data["provider_job_ids"] if persisted else None)
        if len(observations) == 2 and failure is not None:
            raise failure
        return SimpleNamespace(id=f"provider-{len(observations)}")

    device.run.side_effect = submit

    def dispatch(device):
        jobs = [run_circuits(device, object(), shots=10) for _ in range(2)]
        return BenchmarkData(provider_job_ids=[job.id for job in jobs])

    monkeypatch.setattr("metriq_gym.run.setup_device", lambda *_: device)
    monkeypatch.setattr(
        "metriq_gym.run.setup_benchmark", lambda *_: SimpleNamespace(dispatch_handler=dispatch)
    )
    return (dispatch_suite if suite else dispatch_job), args, manager, device, observations


@pytest.mark.parametrize("suite", [False, True])
def test_later_submission_failure_keeps_checkpointed_ids(tmp_path, monkeypatch, suite):
    dispatch, args, manager, device, observations = _dispatch(
        tmp_path, monkeypatch, suite=suite, failure=RuntimeError("second submission rejected")
    )

    dispatch(args, manager)

    assert observations == [[], ["provider-1"]]
    saved = JobManager(manager.jobs_file).get_latest_job()
    assert saved.data["provider_job_ids"] == ["provider-1"]
    assert saved.dispatch_complete is False
    assert saved.failed
    assert saved.error["message"] == "RuntimeError: second submission rejected"
    assert saved.result_data is None
    assert device.run.call_count == 2
    assert (saved.suite_id is not None) == suite


def test_complete_dispatch_replaces_checkpoint_with_benchmark_data(tmp_path, monkeypatch):
    dispatch, args, manager, _, observations = _dispatch(tmp_path, monkeypatch)

    dispatch(args, manager)

    saved = JobManager(manager.jobs_file).get_latest_job()
    assert observations == [[], ["provider-1"]]
    assert saved.dispatch_complete is True
    assert saved.data == {
        "provider_job_ids": ["provider-1", "provider-2"],
        "input_two_qubit_gate_counts": None,
        "transpiled_two_qubit_gate_counts": None,
    }
    assert saved.error is None


def test_interrupt_preserves_accepted_jobs(tmp_path, monkeypatch):
    dispatch, args, manager, _, _ = _dispatch(tmp_path, monkeypatch, failure=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        dispatch(args, manager)

    saved = JobManager(manager.jobs_file).get_latest_job()
    assert saved.data["provider_job_ids"] == ["provider-1"]
    assert saved.dispatch_complete is False


@pytest.mark.parametrize("suite", [False, True])
def test_checkpoint_failure_stops_further_submissions(tmp_path, monkeypatch, suite, capsys):
    dispatch, args, manager, device, _ = _dispatch(tmp_path, monkeypatch, suite=suite)
    if suite:
        suite_file = tmp_path / "suite.json"
        contents = json.loads(suite_file.read_text())
        contents["benchmarks"].append({**contents["benchmarks"][0], "name": "later"})
        suite_file.write_text(json.dumps(contents))

    def fail_replace(*args):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.replace", fail_replace)
    dispatch(args, manager)

    assert device.run.call_count == 1
    output = capsys.readouterr().out
    assert "provider-1" in output
    assert "stopped" in output.lower()
    saved = JobManager(manager.jobs_file).get_latest_job()
    assert saved.dispatch_complete is False


def test_returned_metadata_cannot_drop_accepted_jobs(tmp_path, monkeypatch):
    from metriq_gym.qplatform.submission import run_circuits

    dispatch, args, manager, _, _ = _dispatch(tmp_path, monkeypatch)

    def missing_metadata(device):
        run_circuits(device, object(), shots=10)
        return BenchmarkData(provider_job_ids=[])

    monkeypatch.setattr(
        "metriq_gym.run.setup_benchmark",
        lambda *_: SimpleNamespace(dispatch_handler=missing_metadata),
    )
    dispatch(args, manager)

    saved = JobManager(manager.jobs_file).get_latest_job()
    assert saved.data["provider_job_ids"] == ["provider-1"]
    assert saved.dispatch_complete is False
    assert "omitted accepted" in saved.error["message"]


def test_first_submission_failure_records_empty_incomplete_dispatch(tmp_path, monkeypatch):
    dispatch, args, manager, device, _ = _dispatch(tmp_path, monkeypatch)
    device.run.side_effect = RuntimeError("first submission rejected")

    dispatch(args, manager)

    saved = JobManager(manager.jobs_file).get_latest_job()
    assert saved.data["provider_job_ids"] == []
    assert saved.dispatch_complete is False
    assert saved.failed


def test_initial_record_failure_prevents_any_submission(tmp_path, monkeypatch):
    dispatch, args, manager, device, _ = _dispatch(tmp_path, monkeypatch)
    manager.jobs_file = tmp_path / "missing" / "jobs.jsonl"

    dispatch(args, manager)

    device.run.assert_not_called()


@pytest.mark.parametrize("no_cache", [False, True])
def test_incomplete_poll_reports_status_without_scoring(tmp_path, monkeypatch, no_cache, capsys):
    job = MetriqGymJob(
        id="incomplete",
        job_type=JobType.WIT,
        params={},
        data={"provider_job_ids": ["accepted"]},
        provider_name="local",
        device_name="fake",
        dispatch_time=datetime.now(timezone.utc),
        dispatch_complete=False,
    )
    manager = JobManager(tmp_path / "jobs.jsonl")
    manager.add_job(job)
    provider_job = Mock()
    provider_job.id = "accepted"
    provider_job.status.return_value = JobStatus.COMPLETED
    load = Mock(return_value=provider_job)
    setup = Mock(side_effect=AssertionError("incomplete data must not be scored"))
    monkeypatch.setattr("metriq_gym.run.load_job", load)
    monkeypatch.setattr("metriq_gym.run.setup_benchmark", setup)

    assert fetch_result(job, SimpleNamespace(no_cache=no_cache), manager) is None

    load.assert_called_once()
    setup.assert_not_called()
    provider_job.result.assert_not_called()
    output = capsys.readouterr().out
    assert "accepted: COMPLETED" in output
    assert "incomplete" in output.lower()
    assert job.result_data is None


def test_legacy_job_defaults_to_complete():
    legacy = {
        "id": "legacy",
        "job_type": "WIT",
        "params": {},
        "data": {},
        "provider_name": "local",
        "device_name": "fake",
        "dispatch_time": "2026-01-01T00:00:00+00:00",
    }
    assert MetriqGymJob.deserialize(json.dumps(legacy)).dispatch_complete is True
