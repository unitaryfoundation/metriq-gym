import json
import re
import subprocess
from pathlib import Path

import pytest
from metriq_gym import __version__ as mgym_version
from metriq_gym.upload_paths import minor_series_label


@pytest.fixture(autouse=True)
def store_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MGYM_LOCAL_DB_DIR", str(tmp_path))
    monkeypatch.setenv("MGYM_LOCAL_SIMULATOR_CACHE_DIR", str(tmp_path))


@pytest.mark.e2e
def test_dispatch_and_poll_single_job_on_local_simulator(tmp_path):
    """
    End-to-end test of the CLI workflow for a single job on the local simulator
        1. dispatch   -> returns a Metriq-Gym job_id
        2. poll       -> succeeds immediately for the local simulator
        3. validate   -> JSON result file contains measurement counts
    """

    # ------------------------------------------------------------------
    # 1. Dispatch a tiny benchmark on the local Aer simulator
    # ------------------------------------------------------------------
    example_cfg = Path(__file__).parent.resolve() / "test_benchmark.json"

    subprocess.run(
        [
            "mgym",
            "job",
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

    # ------------------------------------------------------------------
    # 2. Poll the same job and export the JSON payload
    # ------------------------------------------------------------------
    outfile = tmp_path / "test.json"
    poll_cmd = subprocess.run(
        [
            "mgym",
            "job",
            "poll",
            "latest",
            "--json",
            str(outfile),  # temp file to keep the repo clean
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Polling job..." in poll_cmd.stdout

    # ------------------------------------------------------------------
    # 3. Validate the JSON result file
    # ------------------------------------------------------------------
    assert outfile.exists(), "Poll did not write the expected JSON file"

    data = json.loads(outfile.read_text())
    result = data["results"]["accuracy_score"]["value"]
    assert result, "No results found in the JSON file"
    assert result == 1.0, "Expected accuracy score of 1.0 for the local simulator"

    # ------------------------------------------------------------------
    # 4. Dry-run upload (no network/git) and verify file placement
    # ------------------------------------------------------------------
    dry_out = subprocess.run(
        [
            "mgym",
            "job",
            "upload",
            "latest",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "DRY-RUN:" in dry_out.stdout

    # Parse path from DRY-RUN summary and validate file
    line = next(line for line in dry_out.stdout.splitlines() if line.startswith("DRY-RUN:"))
    # format: DRY-RUN: wrote mock file at <path>; would create branch ...
    path_part = line.split(" at ", 1)[1].split(";", 1)[0].strip()
    path = Path(path_part)
    assert path.name.endswith(".json"), "Upload should write a JSON file"

    expected_version_dir = minor_series_label(mgym_version)
    # .../metriq-gym/<version>/local/aer_simulator/<timestamp>_<job_type>_<hash>.json
    assert path.parent.name == "aer_simulator"
    assert path.parent.parent.name == "local"
    assert path.parent.parent.parent.name == expected_version_dir
    assert path.parent.parent.parent.parent.name == "metriq-gym"

    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_qml_kernel_[0-9a-f]{8}\.json", path.name)

    with open(path) as f:
        arr = json.load(f)
        assert isinstance(arr, list) and arr, "Upload payload should be a non-empty JSON array"

    # ------------------------------------------------------------------
    # 5. Clean up
    # ------------------------------------------------------------------
    outfile.unlink(missing_ok=True)
    delete_cmd = subprocess.run(
        [
            "mgym",
            "job",
            "delete",
            "latest",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "deleted successfully" in delete_cmd.stdout, "Failed to delete the job"
