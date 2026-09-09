import json
from datetime import datetime

import pytest

from metriq_gym.benchmarks.benchmark import BenchmarkScore
from metriq_gym.benchmarks.wit import WITResult
from metriq_gym.exporters.json_exporter import JsonExporter
from metriq_gym.job_manager import MetriqGymJob


def test_json_exporter_serializes_none_uncertainty_to_null(metriq_job, tmp_path):
    result = WITResult(expectation_value=BenchmarkScore(value=0.42))
    outfile = tmp_path / "result.json"

    JsonExporter(metriq_job, result).export(str(outfile))

    with open(outfile) as f:
        data = json.load(f)

    assert data["results"]["expectation_value"]["value"] == 0.42
    assert data["results"]["expectation_value"]["uncertainty"] is None


@pytest.mark.parametrize("completed", [True, False])
@pytest.mark.parametrize(
    "dispatch_time, expected",
    [
        ("2026-09-09T14:34:31.123456+00:00", "2026-09-09T14:34:31.123456+00:00"),
        ("2026-09-09T16:34:31.123456+02:00", "2026-09-09T14:34:31.123456+00:00"),
        ("2026-09-09T23:34:31-05:00", "2026-09-10T04:34:31+00:00"),
        ("2026-09-09T00:34:31+05:30", "2026-09-08T19:04:31+00:00"),
        # Legacy jobs stored local wall time without an offset. Use Madrid's
        # offset at dispatch, including winter dates exported during summer.
        ("2026-09-09T16:34:31", "2026-09-09T14:34:31+00:00"),
        ("2026-01-09T16:34:31", "2026-01-09T15:34:31+00:00"),
    ],
)
def test_json_exporter_uses_utc_timestamp(
    metriq_job, tmp_path, local_timezone, completed, dispatch_time, expected
):
    metriq_job.dispatch_time = datetime.fromisoformat(dispatch_time)
    job = MetriqGymJob.deserialize(metriq_job.serialize())
    result = WITResult(expectation_value=BenchmarkScore(value=0.42)) if completed else None
    outfile = tmp_path / "result.json"

    JsonExporter(job, result).export(str(outfile))

    data = json.loads(outfile.read_text())
    assert data["timestamp"] == expected
    assert job.dispatch_time == metriq_job.dispatch_time
