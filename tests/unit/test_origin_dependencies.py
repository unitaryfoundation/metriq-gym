"""Origin SDK availability must not prevent other providers or cached results."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

import metriq_gym.run as run
from metriq_gym.cli import app
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.job_manager import JobManager


@pytest.mark.parametrize("loader", ["load_provider", "load_job"])
@pytest.mark.parametrize("provider", ["origin", " Origin "])
def test_origin_requires_sdk_before_loading_provider_or_job(monkeypatch, loader, provider):
    monkeypatch.delenv("ORIGIN_API_KEY", raising=False)
    monkeypatch.setattr(run, "find_spec", Mock(return_value=None))
    upstream_loader = Mock()
    monkeypatch.setattr(f"qbraid.runtime.{loader}", upstream_loader)

    with pytest.raises(QBraidSetupError, match="pyqpanda3.*Python 3.12–3.14"):
        if loader == "load_provider":
            run.load_provider(provider)
        else:
            run.load_job("origin-job", provider=provider)

    upstream_loader.assert_not_called()


@pytest.mark.parametrize("loader", ["load_provider", "load_job"])
@pytest.mark.parametrize("provider", ["ibm", "aws", "quantinuum", "local", "origin"])
def test_provider_loading_delegates_when_sdk_available_or_unneeded(monkeypatch, loader, provider):
    sdk_lookup = Mock(return_value=object() if provider == "origin" else None)
    monkeypatch.setattr(run, "find_spec", sdk_lookup)
    upstream_loader = Mock()
    monkeypatch.setattr(f"qbraid.runtime.{loader}", upstream_loader)

    if loader == "load_provider":
        assert run.load_provider(provider) is upstream_loader.return_value
        upstream_loader.assert_called_once_with(provider)
    else:
        assert run.load_job("job-1", provider=provider) is upstream_loader.return_value
        upstream_loader.assert_called_once_with("job-1", provider=provider)

    if provider != "origin":
        sdk_lookup.assert_not_called()


def test_origin_dispatch_reports_missing_sdk_without_creating_job(monkeypatch, caplog):
    monkeypatch.delenv("ORIGIN_API_KEY", raising=False)
    monkeypatch.setattr(run, "find_spec", Mock(return_value=None))
    upstream_loader = Mock()
    monkeypatch.setattr("qbraid.runtime.load_provider", upstream_loader)
    manager = Mock()

    run.dispatch_job(
        SimpleNamespace(provider="origin", device="WK_C102_400", config="unused.json"), manager
    )

    assert "pyqpanda3" in caplog.text
    assert "Python 3.12–3.14" in caplog.text
    upstream_loader.assert_not_called()
    manager.add_job.assert_not_called()


@pytest.mark.parametrize("command", ["job", "suite"])
@pytest.mark.parametrize("cache_mode", ["missing", "cached", "refresh"])
def test_origin_poll_uses_cache_or_reports_missing_sdk(
    monkeypatch, tmp_path, metriq_job, command, cache_mode
):
    monkeypatch.setenv("MGYM_LOCAL_DB_DIR", str(tmp_path))
    monkeypatch.delenv("ORIGIN_API_KEY", raising=False)
    sdk_lookup = Mock(return_value=None)
    monkeypatch.setattr(run, "find_spec", sdk_lookup)
    upstream_loader = Mock()
    monkeypatch.setattr("qbraid.runtime.load_job", upstream_loader)
    cached_result = {"expectation_value": {"value": 0.25, "uncertainty": None}}
    job = replace(
        metriq_job,
        provider_name="origin",
        device_name="WK_C102_400",
        params={"benchmark_name": "WIT", "num_qubits": 6, "shots": 100},
        suite_id="origin-suite",
        result_data=None if cache_mode == "missing" else cached_result,
    )
    manager = JobManager()
    manager.add_job(job)
    args = [command, "poll", job.id if command == "job" else job.suite_id]
    if cache_mode == "refresh":
        args.append("--no-cache")

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    upstream_loader.assert_not_called()
    if cache_mode == "cached":
        assert "[Cached result data]" in result.output
        sdk_lookup.assert_not_called()
    else:
        assert "Could not fetch status/results" in result.output
        assert "pyqpanda3" in result.output
        assert "Python 3.12–3.14" in result.output
        assert "not yet completed" not in result.output
    stored = JobManager().get_job(job.id)
    assert stored.error is None
    assert stored.result_data == job.result_data
