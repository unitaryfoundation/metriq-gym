import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from metriq_gym.cli import app
from metriq_gym.job_manager import JobManager


@pytest.fixture
def suite_manager(metriq_job, monkeypatch, tmp_path):
    monkeypatch.setenv("MGYM_LOCAL_DB_DIR", str(tmp_path))
    manager = JobManager()
    for index, (width, value, uncertainty) in enumerate([(6, 0.25, None), (7, 0.75, 0.02)]):
        manager.add_job(
            replace(
                metriq_job,
                id=f"job-{index}",
                suite_id="suite-1",
                suite_name="test-suite",
                params={"benchmark_name": "WIT", "num_qubits": width, "shots": 100},
                dispatch_time=datetime(2026, 9, 10, 12, index, tzinfo=timezone.utc),
                runtime_seconds=index + 0.5,
                result_data={"expectation_value": {"value": value, "uncertainty": uncertainty}},
            )
        )
    return manager


def test_suite_poll_json_exports_cached_results_and_metadata(suite_manager, tmp_path):
    outfile = tmp_path / "suite results.json"

    result = CliRunner().invoke(app, ["suite", "poll", "suite-1", "--json", str(outfile)])

    assert result.exit_code == 0, result.output
    assert f"Results exported to {outfile}" in result.output
    records = json.loads(outfile.read_text())
    assert len(records) == 2
    for record, job, value, uncertainty in zip(
        records, suite_manager.get_jobs(), [0.25, 0.75], [None, 0.02]
    ):
        assert record["results"] == {
            "expectation_value": {"value": value, "uncertainty": uncertainty},
            "score": {"value": value, "uncertainty": uncertainty},
        }
        assert record["params"] == job.params
        assert record["platform"] == job.platform
        assert record["app_version"] == job.app_version
        assert record["job_type"] == "WIT"
        assert record["suite_id"] == "suite-1"
        assert record["suite"] == {"id": "suite-1", "name": "test-suite"}
        assert record["timestamp"] == job.dispatch_time.isoformat()
        assert record["runtime_seconds"] == job.runtime_seconds


def test_suite_poll_json_skips_failed_jobs_without_mixing_results(suite_manager, tmp_path):
    failed = suite_manager.get_job("job-0")
    failed.result_data = None
    failed.record_error("dispatch", "Submission failed")
    suite_manager.update_job(failed)
    outfile = tmp_path / "suite.json"

    result = CliRunner().invoke(app, ["suite", "poll", "suite-1", "--json", str(outfile)])

    assert result.exit_code == 0, result.output
    assert "failed; skipping" in result.output
    records = json.loads(outfile.read_text())
    assert len(records) == 1
    assert records[0]["params"]["num_qubits"] == 7
    assert records[0]["results"]["score"] == {"value": 0.75, "uncertainty": 0.02}
    assert records[0]["timestamp"] == "2026-09-10T12:01:00+00:00"


@pytest.mark.parametrize("existing_file", [False, True])
def test_suite_poll_json_waits_for_pending_jobs(
    suite_manager, tmp_path, monkeypatch, existing_file
):
    from metriq_gym.run import fetch_result

    pending = suite_manager.get_job("job-1")
    pending.result_data = None
    suite_manager.update_job(pending)
    monkeypatch.setattr(
        "metriq_gym.run.fetch_result",
        lambda job, args, manager: (
            None if job.id == pending.id else fetch_result(job, args, manager)
        ),
    )
    outfile = tmp_path / "suite.json"
    if existing_file:
        outfile.write_text("previous export")

    result = CliRunner().invoke(app, ["suite", "poll", "suite-1", "--json", str(outfile)])

    assert result.exit_code == 0, result.output
    assert "not yet completed" in result.output
    if existing_file:
        assert outfile.read_text() == "previous export"
    else:
        assert not outfile.exists()


def test_suite_poll_json_does_not_export_when_all_jobs_failed(suite_manager, tmp_path):
    for job in suite_manager.get_jobs():
        job.result_data = None
        job.record_error("dispatch", "Submission failed")
        suite_manager.update_job(job)
    outfile = tmp_path / "suite.json"

    result = CliRunner().invoke(app, ["suite", "poll", "suite-1", "--json", str(outfile)])

    assert result.exit_code == 0, result.output
    assert not outfile.exists()
    assert "Results exported" not in result.output
