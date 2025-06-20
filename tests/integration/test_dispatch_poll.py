import json
import re
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def store_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MGR_LOCAL_JOB_DIR", str(tmp_path))


@pytest.mark.e2e
def test_dispatch_and_poll_local_simulator(tmp_path):
    """
    End-to-end test of the CLI workflow:

        1. dispatch   -> returns a Metriq-Gym job_id
        2. poll       -> succeeds immediately for the local simulator
        3. validate   -> JSON result file contains measurement counts
    """

    # ------------------------------------------------------------------
    # 1. Dispatch a tiny benchmark on the local Aer simulator
    # ------------------------------------------------------------------
    example_cfg = (
        Path(__file__).resolve().parents[2]  # repo root
        / "metriq_gym/schemas/examples/qml_kernel.example.json"
    )

    dispatch = subprocess.run(
        [
            "mgym",
            "dispatch",
            str(example_cfg),
            "-p",
            "local",
            "-d",
            "aer_simulator",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Extract the UUID that the CLI prints as:
    #   "dispatched with ID: <uuid>"
    m = re.search(r"dispatched with ID: ([0-9a-f-]{36})", dispatch.stdout)
    assert m, f"Could not parse job_id from:\n{dispatch.stdout}"
    job_id = m.group(1)

    # ------------------------------------------------------------------
    # 2. Poll the same job and export the JSON payload
    # ------------------------------------------------------------------
    outfile = tmp_path / "test.json"
    poll = subprocess.run(
        [
            "mgym",
            "poll",
            "--job_id",
            job_id,
            "--json",
            str(outfile),  # keep the repo clean
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Polling job..." in poll.stdout

    # ------------------------------------------------------------------
    # 3. Validate the JSON result file
    # ------------------------------------------------------------------
    assert outfile.exists(), "Poll did not write the expected JSON file"

    data = json.loads(outfile.read_text())
    result = data["results"]["accuracy_score"]
    assert result, "No results found in the JSON file"
    assert result == 1.0, "Expected accuracy score of 1.0 for the local simulator"
