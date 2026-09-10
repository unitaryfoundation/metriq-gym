import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from metriq_gym.job_manager import JobManager


@pytest.fixture(autouse=True)
def store_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MGYM_LOCAL_DB_DIR", str(tmp_path))
    monkeypatch.setenv("MGYM_LOCAL_SIMULATOR_CACHE_DIR", str(tmp_path))


@pytest.mark.e2e
@pytest.mark.parametrize("local_timezone", ["Europe/Madrid", "America/New_York"], indirect=True)
def test_dispatch_and_poll_suite_on_local_simulator(tmp_path, local_timezone):
    """
    End-to-end test of the CLI workflow for a suite with two jobs on the local simulator
        1. dispatch   -> returns a Metriq-Gym suite_id and two job_ids
        2. poll       -> displays results and exports both jobs to JSON
        3. upload     -> dry-run payload matches the JSON polling export
    """

    # ------------------------------------------------------------------
    # 1. Dispatch a suite with two benchmarks on the local Aer simulator
    # ------------------------------------------------------------------
    example_suite_cfg = Path(__file__).parent.resolve() / "test_suite.json"
    before_dispatch = datetime.now(timezone.utc)

    dispatch_cmd = subprocess.run(
        [
            "mgym",
            "suite",
            "dispatch",
            str(example_suite_cfg),
            "-p",
            "local",
            "-d",
            "aer_simulator",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    after_dispatch = datetime.now(timezone.utc)
    assert "Dispatch complete for suite" in dispatch_cmd.stdout

    # Extract suite_id from output (assumes suite_id is printed in stdout)
    suite_id = None
    for line in dispatch_cmd.stdout.splitlines():
        if "Suite ID" in line:
            suite_id = line.split()[-1].strip(".")
            break
    assert suite_id, "Suite ID not found in dispatch output"

    jobs = JobManager().get_jobs_by_suite_id(suite_id)
    assert len(jobs) == 2
    for job in jobs:
        assert job.dispatch_time.tzinfo is timezone.utc
        assert before_dispatch <= job.dispatch_time <= after_dispatch

    # ------------------------------------------------------------------
    # 2. Poll the suite
    # ------------------------------------------------------------------
    poll_cmd = subprocess.run(
        ["mgym", "suite", "poll", suite_id],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Suite Results" in poll_cmd.stdout, "Suite results not found in poll output"

    json_path = tmp_path / "suite-results.json"
    subprocess.run(
        ["mgym", "suite", "poll", suite_id, "--json", str(json_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    poll_records = json.loads(json_path.read_text())
    assert isinstance(poll_records, list) and len(poll_records) == 2

    # ------------------------------------------------------
    # 3. Dry-run suite upload (single PR, no network/git)
    # ------------------------------------------------------
    upload_out = subprocess.run(
        [
            "mgym",
            "suite",
            "upload",
            suite_id,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "DRY-RUN:" in upload_out.stdout
    # Parse path from DRY-RUN summary and validate file has multiple records
    line = next(line for line in upload_out.stdout.splitlines() if line.startswith("DRY-RUN:"))
    path_part = line.split(" at ", 1)[1].split(";", 1)[0].strip()

    with open(path_part) as f:
        arr = json.load(f)
    assert arr == poll_records
    assert [record["timestamp"] for record in arr] == [
        job.dispatch_time.isoformat() for job in jobs
    ]
    assert all(record["timestamp"].endswith("+00:00") for record in arr)
    expected_prefix = jobs[0].dispatch_time.strftime("%Y-%m-%d_%H-%M-%S")
    assert Path(path_part).name.startswith(f"{expected_prefix}_local_sim_suite_")

    # Re-export the same suite using the naive local timestamps stored by older
    # versions. Every record and the filename must still identify the same UTC time.
    job_manager = JobManager()
    for job in job_manager.get_jobs_by_suite_id(suite_id):
        job.dispatch_time = job.dispatch_time.astimezone().replace(tzinfo=None)
        job_manager.update_job(job)
    legacy_json_path = tmp_path / "legacy-suite-results.json"
    subprocess.run(
        ["mgym", "suite", "poll", suite_id, "--json", str(legacy_json_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(legacy_json_path.read_text()) == poll_records
    legacy_upload = subprocess.run(
        ["mgym", "suite", "upload", suite_id, "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    legacy_line = next(
        line for line in legacy_upload.stdout.splitlines() if line.startswith("DRY-RUN:")
    )
    legacy_path = Path(legacy_line.split(" at ", 1)[1].split(";", 1)[0].strip())
    assert json.loads(legacy_path.read_text()) == arr
    assert legacy_path.name == Path(path_part).name

    # ------------------------------------------------------
    # 4. Delete the suite to clean up
    # ------------------------------------------------------
    delete_cmd = subprocess.run(
        ["mgym", "suite", "delete", suite_id],
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"All jobs for suite ID {suite_id} deleted successfully" in delete_cmd.stdout
