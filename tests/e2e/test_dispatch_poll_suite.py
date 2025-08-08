import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def store_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MGR_LOCAL_JOB_DIR", str(tmp_path))


@pytest.mark.e2e
def test_dispatch_and_poll_suite_on_local_simulator(tmp_path):
    """
    End-to-end test of the CLI workflow for a suite with two jobs on the local simulator
        1. dispatch   -> returns a Metriq-Gym suite_id and two job_ids
        2. poll       -> succeeds immediately for the local simulator
        3. validate   -> JSON result file contains results for both jobs
    """

    # ------------------------------------------------------------------
    # 1. Dispatch a suite with two benchmarks on the local Aer simulator
    # ------------------------------------------------------------------
    example_suite_cfg = Path(__file__).parent.resolve() / "test_suite.json"

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
    assert "Dispatch complete for suite" in dispatch_cmd.stdout

    # Extract suite_id from output (assumes suite_id is printed in stdout)
    suite_id = None
    for line in dispatch_cmd.stdout.splitlines():
        if "Suite ID" in line:
            suite_id = line.split()[-1].strip(".")
            break
    assert suite_id, "Suite ID not found in dispatch output"

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

    # ------------------------------------------------------
    # 3. Delete the suite to clean up
    # ------------------------------------------------------
    delete_cmd = subprocess.run(
        ["mgym", "suite", "delete", suite_id],
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"All jobs for suite ID {suite_id} deleted successfully" in delete_cmd.stdout
