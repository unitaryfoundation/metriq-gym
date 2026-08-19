"""Tests for failure capture on jobs and non-completed outcome uploads.

Mirrors the metriq-data "Record outcomes" contract: a failed attempt is kept on the
job record (verbatim error) and can be uploaded as an ``outcome`` record instead of
silently disappearing.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from qbraid.runtime import JobStatus

from metriq_gym.benchmarks.benchmark import BenchmarkData, BenchmarkResult, BenchmarkScore
from metriq_gym.constants import JobType
from metriq_gym.exporters.dict_exporter import DictExporter
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.run import (
    _resolve_upload_outcome,
    dispatch_job,
    fetch_result,
    upload_job,
    upload_suite,
)


def _job(*, error=None, result_data=None, provider_job_ids=("pj-1",), suite_id=None):
    return MetriqGymJob(
        id="job-1",
        job_type=JobType.WIT,
        params={"benchmark_name": "WIT", "shots": 10},
        data={"provider_job_ids": list(provider_job_ids)},
        provider_name="aws",
        device_name="arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
        dispatch_time=datetime(2026, 8, 7, 12, 0, 0),
        suite_id=suite_id,
        result_data=result_data,
        error=error,
    )


class DummyResult(BenchmarkResult):
    value: int

    def compute_score(self):
        return BenchmarkScore(value=float(self.value), uncertainty=None)


@dataclass
class DummyJobData(BenchmarkData):
    provider_job_ids: list[str]


class FailedQuantumJob:
    def __init__(self, job_id, status=JobStatus.FAILED):
        self.id = job_id
        self._status = status

    def status(self):
        return self._status


# --- job record -----------------------------------------------------------------


def test_record_error_from_exception_keeps_type_and_message():
    job = _job()
    job.record_error("dispatch", ValueError("Uses 'barrier' which is not supported"))
    assert job.error["source"] == "dispatch"
    assert job.error["message"] == "ValueError: Uses 'barrier' which is not supported"
    datetime.fromisoformat(job.error["timestamp"])
    assert job.failed


def test_failed_is_false_once_results_exist():
    job = _job(error={"source": "poll", "message": "boom", "timestamp": "t"})
    assert job.failed
    job.result_data = {"value": 1}
    assert not job.failed


def test_error_round_trips_through_serialization_and_loads_in_job_manager(tmp_path):
    job = _job()
    job.record_error("poll", "pj-1: FAILED - compilation failed")
    jm = JobManager(jobs_file=tmp_path / "jobs.jsonl")
    jm.add_job(job)

    reloaded = JobManager(jobs_file=tmp_path / "jobs.jsonl").get_job("job-1")
    assert reloaded.error == job.error
    assert reloaded.failed


def test_str_tolerates_missing_provider_job_ids():
    job = _job(provider_job_ids=())
    job.data = {}
    assert "error" in str(job)


# --- exporter ------------------------------------------------------------------


def test_completed_record_has_no_outcome_fields():
    record = DictExporter(_job(), DummyResult(value=3)).export()
    assert "outcome" not in record
    assert "outcome_detail" not in record
    assert record["results"]["value"] == 3


def test_failed_job_exports_error_outcome_with_captured_message():
    job = _job(error={"source": "poll", "message": "pj-1: FAILED - boom", "timestamp": "t"})
    record = DictExporter(job, None).export()
    assert record["outcome"] == "error"
    assert record["results"] is None
    assert record["outcome_detail"] == {"error_message": "pj-1: FAILED - boom", "source": "poll"}
    # Platform/params machinery is unchanged so the instance is identifiable downstream.
    assert record["platform"]["provider"] == "aws"
    assert record["platform"]["device"] == "rigetti_cepheus-1-108q"
    assert record["job_type"] == "WIT"


def test_human_outcome_with_reason_and_evidence():
    job = _job(error={"source": "dispatch", "message": "ValidationException: x", "timestamp": "t"})
    record = DictExporter(
        job, None, outcome="unsupported", outcome_reason="Compiler rejects 100q circuits"
    ).export()
    assert record["outcome"] == "unsupported"
    assert record["outcome_detail"] == {
        "reason": "Compiler rejects 100q circuits",
        "error_message": "ValidationException: x",
        "source": "dispatch",
    }
    assert record["results"] is None


def test_outcome_without_any_detail_emits_null_detail():
    record = DictExporter(_job(), None, outcome="not_applicable").export()
    assert record["outcome"] == "not_applicable"
    assert record["outcome_detail"] is None


def test_explicit_outcome_drops_results_even_if_result_given():
    record = DictExporter(_job(), DummyResult(value=3), outcome="unsupported").export()
    assert record["outcome"] == "unsupported"
    assert record["results"] is None


def test_unknown_outcome_rejected():
    with pytest.raises(ValueError, match="Unknown outcome"):
        DictExporter(_job(), None, outcome="exploded")


def test_outcome_record_is_json_serializable():
    job = _job(error={"source": "poll", "message": "boom", "timestamp": "t"})
    json.dumps(DictExporter(job, None).export() | {"params": job.params})


# --- fetch_result ---------------------------------------------------------------


def _patch_fetch(monkeypatch, quantum_job_factory):
    import metriq_gym.run as run_mod

    monkeypatch.setattr(run_mod, "setup_benchmark_result_class", lambda *_: DummyResult)
    monkeypatch.setattr(run_mod, "setup_job_data_class", lambda *_: DummyJobData)
    monkeypatch.setattr(run_mod, "setup_benchmark", lambda *_, **__: MagicMock())
    monkeypatch.setattr(run_mod, "load_job", lambda job_id, **__: quantum_job_factory(job_id))
    monkeypatch.setattr(run_mod, "validate_and_create_model", lambda params: params)


def test_fetch_result_records_provider_failure_on_job(monkeypatch, tmp_path, capsys):
    job = _job()
    jm = JobManager(jobs_file=tmp_path / "jobs.jsonl")
    jm.add_job(job)
    _patch_fetch(monkeypatch, lambda jid: FailedQuantumJob(jid))
    monkeypatch.setattr(
        "metriq_gym.run.failed_jobs_summary", lambda qjobs: "pj-1: FAILED - compilation failed"
    )
    args = SimpleNamespace(no_cache=False, include_raw=False)

    assert fetch_result(job, args, jm) is None

    assert job.error["source"] == "poll"
    assert job.error["message"] == "pj-1: FAILED - compilation failed"
    # Persisted, not just in memory.
    assert JobManager(jobs_file=tmp_path / "jobs.jsonl").get_job("job-1").failed
    out = capsys.readouterr().out
    assert "Job failed. Provider reported:" in out
    assert "compilation failed" in out


def test_fetch_result_pending_job_does_not_record_error(monkeypatch, tmp_path, capsys):
    job = _job()
    jm = JobManager(jobs_file=tmp_path / "jobs.jsonl")
    jm.add_job(job)
    _patch_fetch(monkeypatch, lambda jid: FailedQuantumJob(jid, JobStatus.QUEUED))
    monkeypatch.setattr(
        "metriq_gym.run.job_status",
        lambda task: SimpleNamespace(status=JobStatus.QUEUED, queue_position=None),
    )
    args = SimpleNamespace(no_cache=False, include_raw=False)

    assert fetch_result(job, args, jm) is None
    assert job.error is None
    assert "not yet completed" in capsys.readouterr().out


def test_fetch_result_short_circuits_dispatch_failure(monkeypatch, tmp_path, capsys):
    job = _job(
        provider_job_ids=(),
        error={"source": "dispatch", "message": "ValidationException: barrier", "timestamp": "t"},
    )
    jm = JobManager(jobs_file=tmp_path / "jobs.jsonl")
    jm.add_job(job)
    load_job = MagicMock(side_effect=AssertionError("must not poll provider"))
    _patch_fetch(monkeypatch, load_job)
    args = SimpleNamespace(no_cache=False, include_raw=False)

    assert fetch_result(job, args, jm) is None
    assert "Job failed at dispatch" in capsys.readouterr().out
    load_job.assert_not_called()


# --- dispatch -----------------------------------------------------------------


def test_dispatch_job_records_failed_attempt(monkeypatch, tmp_path, capsys):
    class Params(BaseModel):
        benchmark_name: str = "WIT"
        shots: int = 10

    jm = JobManager(jobs_file=tmp_path / "jobs.jsonl")
    device = SimpleNamespace(id="arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1", num_qubits=12)
    registry = MagicMock()
    registry.get_available_benchmarks.return_value = ["WIT"]
    handler = MagicMock()
    handler.dispatch_handler.side_effect = ValueError("Uses 'barrier' which is not supported")

    monkeypatch.setattr("os.path.exists", lambda *_: True)
    monkeypatch.setattr("metriq_gym.run.setup_device", lambda *_: device)
    monkeypatch.setattr("metriq_gym.run.load_and_validate", lambda *_: Params())
    monkeypatch.setattr("metriq_gym.run._lazy_registry", lambda: registry)
    monkeypatch.setattr("metriq_gym.run.setup_benchmark", lambda *_, **__: handler)
    monkeypatch.setattr("metriq_gym.qplatform.device.normalized_metadata", lambda *_: {})

    args = SimpleNamespace(config="wit.json", provider="aws", device=device.id)
    dispatch_job(args, jm)

    out = capsys.readouterr().out
    assert "failed to dispatch" in out
    assert "Recorded as failed" in out
    (job,) = jm.get_jobs()
    assert job.failed
    assert job.error["source"] == "dispatch"
    assert "Uses 'barrier'" in job.error["message"]
    assert job.data["provider_job_ids"] == []
    assert job.params == {"benchmark_name": "WIT", "shots": 10}


# --- upload -------------------------------------------------------------------


def test_resolve_upload_outcome_failed_job_defaults_to_error():
    job = _job(error={"source": "poll", "message": "m", "timestamp": "t"})
    args = SimpleNamespace(outcome=None, reason=None)
    assert _resolve_upload_outcome(args, job, has_result=False) == ("error", None)


def test_resolve_upload_outcome_pending_job_refused(capsys):
    args = SimpleNamespace(outcome=None, reason=None)
    assert _resolve_upload_outcome(args, _job(), has_result=False) is None
    assert "not yet completed" in capsys.readouterr().out


def test_resolve_upload_outcome_completed_job_is_plain_upload():
    args = SimpleNamespace(outcome=None, reason=None)
    assert _resolve_upload_outcome(args, _job(), has_result=True) == (None, None)


def test_resolve_upload_outcome_refuses_to_reclassify_completed_job(capsys):
    args = SimpleNamespace(outcome="unsupported", reason="r")
    assert _resolve_upload_outcome(args, _job(), has_result=True) is None
    assert "refusing" in capsys.readouterr().out


@pytest.mark.parametrize("outcome", ["unsupported", "not_applicable"])
def test_resolve_upload_outcome_human_outcomes_require_reason(outcome, capsys):
    job = _job(error={"source": "poll", "message": "m", "timestamp": "t"})
    args = SimpleNamespace(outcome=outcome, reason=None)
    assert _resolve_upload_outcome(args, job, has_result=False) is None
    assert "requires --reason" in capsys.readouterr().out
    args.reason = "because"
    assert _resolve_upload_outcome(args, job, has_result=False) == (outcome, "because")


def test_resolve_upload_outcome_rejects_unknown_value(capsys):
    args = SimpleNamespace(outcome="exploded", reason=None)
    assert _resolve_upload_outcome(args, _job(), has_result=False) is None
    assert "--outcome must be one of" in capsys.readouterr().out


def test_upload_job_failed_job_writes_error_outcome_record(monkeypatch, tmp_path, capsys):
    job = _job(
        provider_job_ids=(),
        error={"source": "dispatch", "message": "ValidationException: barrier", "timestamp": "t"},
    )
    jm = JobManager(jobs_file=tmp_path / "jobs.jsonl")
    jm.add_job(job)
    monkeypatch.setattr("metriq_gym.run.setup_job_data_class", lambda *_: DummyJobData)
    monkeypatch.setattr("metriq_gym.run.setup_benchmark_result_class", lambda *_: DummyResult)

    captured = {}

    class FakeExporter:
        def __init__(self, job, result):
            captured["result"] = result

        def export(self, **kwargs):
            captured.update(kwargs)
            return "DRY-RUN: ok"

    monkeypatch.setattr("metriq_gym.exporters.github_pr_exporter.GitHubPRExporter", FakeExporter)

    args = SimpleNamespace(
        job_id="job-1",
        repo="owner/repo",
        dry_run=True,
        no_cache=False,
        include_raw=False,
        outcome="unsupported",
        reason="Device rejects barriers",
    )
    upload_job(args, jm)

    out = capsys.readouterr().out
    assert "Uploading as 'unsupported' outcome record" in out
    assert "DRY-RUN: ok" in out
    payload = captured["payload"]
    assert payload["outcome"] == "unsupported"
    assert payload["results"] is None
    assert payload["outcome_detail"]["reason"] == "Device rejects barriers"
    assert payload["outcome_detail"]["error_message"] == "ValidationException: barrier"
    assert payload["params"] == job.params
    assert captured["pr_title"].endswith("(unsupported)")
    assert captured["result"] is None


def test_upload_suite_includes_failed_jobs_as_error_outcomes(monkeypatch, tmp_path, capsys):
    ok_job = _job(suite_id="suite-1", result_data={"value": 5})
    failed = _job(
        provider_job_ids=(),
        suite_id="suite-1",
        error={"source": "dispatch", "message": "boom", "timestamp": "t"},
    )
    failed.id = "job-2"
    jm = JobManager(jobs_file=tmp_path / "jobs.jsonl")
    jm.add_job(ok_job)
    jm.add_job(failed)
    monkeypatch.setattr("metriq_gym.run.setup_job_data_class", lambda *_: DummyJobData)
    monkeypatch.setattr("metriq_gym.run.setup_benchmark_result_class", lambda *_: DummyResult)

    captured = {}

    class FakeExporter:
        def __init__(self, job, result):
            pass

        def export(self, **kwargs):
            captured.update(kwargs)
            return "DRY-RUN: ok"

    monkeypatch.setattr("metriq_gym.exporters.github_pr_exporter.GitHubPRExporter", FakeExporter)

    args = SimpleNamespace(
        suite_id="suite-1", repo="owner/repo", dry_run=True, no_cache=False, include_raw=False
    )
    upload_suite(args, jm)

    assert "will be uploaded as an 'error' outcome" in capsys.readouterr().out
    records = captured["payload"]
    assert len(records) == 2
    assert "outcome" not in records[0]
    assert records[0]["results"]["value"] == 5
    assert records[1]["outcome"] == "error"
    assert records[1]["results"] is None
    assert records[1]["outcome_detail"]["error_message"] == "boom"
