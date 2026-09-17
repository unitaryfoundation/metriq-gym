import json

import pytest

from metriq_gym.dashboard.server import db_path, wire_jobs


@pytest.fixture
def dashboard_db(monkeypatch, tmp_path):
    monkeypatch.setenv("MGYM_LOCAL_DB_DIR", str(tmp_path))
    return db_path()


@pytest.mark.parametrize(
    "local_timezone, legacy_timestamp, utc_timestamp",
    [
        ("Europe/Madrid", "2026-09-09T16:00:00", "2026-09-09T14:01:00+00:00"),
        ("Europe/Madrid", "2026-01-09T16:00:00", "2026-01-09T15:01:00+00:00"),
        ("America/New_York", "2026-09-09T10:00:00", "2026-09-09T14:01:00+00:00"),
        ("America/New_York", "2026-01-09T09:00:00", "2026-01-09T14:01:00+00:00"),
    ],
    indirect=["local_timezone"],
)
def test_wire_jobs_sorts_mixed_dispatch_times_chronologically(
    dashboard_db, local_timezone, legacy_timestamp, utc_timestamp
):
    month = utc_timestamp[5:7]
    records = [
        {"id": "legacy", "dispatch_time": legacy_timestamp},
        {"id": "new-utc", "dispatch_time": utc_timestamp},
        # These offsets cross a date boundary; raw string order would put the
        # older record ahead of the newer one.
        {"id": "old-offset", "dispatch_time": f"2026-{month}-10T00:00:00+14:00"},
        {"id": "new-offset", "dispatch_time": f"2026-{month}-09T23:00:00-12:00"},
    ]
    dashboard_db.write_text("".join(json.dumps(record) + "\n" for record in records))

    jobs = wire_jobs()

    assert [job["id"] for job in jobs] == ["new-offset", "new-utc", "legacy", "old-offset"]
    # Sorting must not rewrite the stored timestamps or the API values.
    original_times = {record["id"]: record["dispatch_time"] for record in records}
    assert {job["id"]: job["dispatch_time"] for job in jobs} == original_times
    assert [json.loads(line) for line in dashboard_db.read_text().splitlines()] == records


def test_wire_jobs_keeps_invalid_dispatch_times_at_the_end(dashboard_db, local_timezone):
    records = [
        {"id": "invalid", "dispatch_time": "not-a-date"},
        {"id": "missing"},
        {"id": "null", "dispatch_time": None},
        {"id": "numeric", "dispatch_time": 123},
        {"id": "empty", "dispatch_time": ""},
        {"id": "valid", "dispatch_time": "2026-09-09T14:01:00Z"},
    ]
    dashboard_db.write_text("".join(json.dumps(record) + "\n" for record in records))

    jobs = wire_jobs()

    assert [job["id"] for job in jobs] == [
        "valid",
        "invalid",
        "missing",
        "null",
        "numeric",
        "empty",
    ]
