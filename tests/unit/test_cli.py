from datetime import datetime
from tabulate import tabulate
import pytest
from unittest.mock import MagicMock
from metriq_gym.cli import LIST_JOBS_HEADERS, list_jobs
from metriq_gym.constants import JobType
from metriq_gym.job_manager import JobManager, MetriqGymJob


@pytest.fixture
def mock_job_manager():
    """Fixture to provide a mocked JobManager."""
    return MagicMock(spec=JobManager)


def test_list_jobs_all(capsys):
    """Test listing all jobs without filters."""
    # Mock jobs
    mock_jobs = [
        MetriqGymJob(
            id="1234",
            device_name="ibmq_qasm_simulator",
            provider_name="ibm",
            job_type=JobType.QUANTUM_VOLUME,
            dispatch_time=datetime.fromisoformat("2021-09-01T12:00:00"),
            params={},
            data={},
        ),
        MetriqGymJob(
            id="5678",
            device_name="ionq_simulator",
            provider_name="ionq",
            job_type=JobType.QUANTUM_VOLUME,
            dispatch_time=datetime.fromisoformat("2021-09-02T12:00:00"),
            params={},
            data={},
        ),
    ]

    list_jobs(mock_jobs, show_index=False, show_suite_id=False)

    # Capture the output
    captured = capsys.readouterr()

    # Expected output using tabulate
    table = [
        ["1234", "ibm", "ibmq_qasm_simulator", "Quantum Volume", None, "2021-09-01T12:00:00"],
        ["5678", "ionq", "ionq_simulator", "Quantum Volume", None, "2021-09-02T12:00:00"],
    ]
    expected_output = tabulate(table, headers=LIST_JOBS_HEADERS, tablefmt="grid") + "\n"

    assert captured.out == expected_output


def test_list_jobs_prefers_max_qubits_for_qft(capsys):
    mock_jobs = [
        MetriqGymJob(
            id="qft1",
            device_name="local_sim",
            provider_name="local",
            job_type=JobType.QUANTUM_FOURIER_TRANSFORM,
            dispatch_time=datetime.fromisoformat("2021-09-01T12:00:00"),
            params={"max_qubits": 6},
            data={},
        )
    ]

    list_jobs(mock_jobs, show_index=False, show_suite_id=False)
    captured = capsys.readouterr()

    table = [["qft1", "local", "local_sim", "Quantum Fourier Transform", 6, "2021-09-01T12:00:00"]]
    expected_output = tabulate(table, headers=LIST_JOBS_HEADERS, tablefmt="grid") + "\n"
    assert captured.out == expected_output


def test_list_jobs_uses_width_alias_for_num_qubits(capsys):
    mock_jobs = [
        MetriqGymJob(
            id="mc1",
            device_name="local_sim",
            provider_name="local",
            job_type=JobType.MIRROR_CIRCUITS,
            dispatch_time=datetime.fromisoformat("2021-09-01T12:00:00"),
            params={"width": 11, "max_qubits": 20},
            data={},
        )
    ]

    list_jobs(mock_jobs, show_index=False, show_suite_id=False)
    captured = capsys.readouterr()

    table = [["mc1", "local", "local_sim", "Mirror Circuits", 11, "2021-09-01T12:00:00"]]
    expected_output = tabulate(table, headers=LIST_JOBS_HEADERS, tablefmt="grid") + "\n"
    assert captured.out == expected_output


def test_list_jobs_uses_qubits_alias_for_num_qubits(capsys):
    mock_jobs = [
        MetriqGymJob(
            id="q2",
            device_name="local_sim",
            provider_name="local",
            job_type=JobType.WIT,
            dispatch_time=datetime.fromisoformat("2021-09-01T12:00:00"),
            params={"qubits": 9},
            data={},
        )
    ]

    list_jobs(mock_jobs, show_index=False, show_suite_id=False)
    captured = capsys.readouterr()

    table = [["q2", "local", "local_sim", "WIT", 9, "2021-09-01T12:00:00"]]
    expected_output = tabulate(table, headers=LIST_JOBS_HEADERS, tablefmt="grid") + "\n"
    assert captured.out == expected_output


def test_list_jobs_no_jobs(capsys):
    """Test listing jobs when no jobs are recorded."""
    # Mock no jobs
    mock_jobs = []

    list_jobs(mock_jobs, show_index=False)

    # Capture the output
    captured = capsys.readouterr()

    # Verify the printed output
    assert captured.out == "No jobs found.\n"


def test_job_poll_include_raw_flag():
    """Test that --include-raw flag is passed correctly in job_poll."""
    from typer.testing import CliRunner
    from metriq_gym.cli import app
    from unittest.mock import patch

    runner = CliRunner()

    with patch("metriq_gym.cli.JobManager") as mock_jm_class:
        with patch("metriq_gym.run.fetch_result") as mock_fetch:
            # Setup mock job manager
            mock_jm = MagicMock()
            mock_jm_class.return_value = mock_jm
            mock_job = MagicMock()
            mock_job.id = "test-job"
            mock_jm.get_latest_job.return_value = mock_job

            # Mock fetch_result to return a valid output
            mock_result = MagicMock()
            mock_fetch.return_value = MagicMock(
                result=mock_result, raw_counts=None, from_cache=False
            )

            # Run CLI with --include-raw
            runner.invoke(app, ["job", "poll", "latest", "--include-raw"])

            # Verify fetch_result was called with args that have include_raw=True
            assert mock_fetch.called
            args = mock_fetch.call_args[0][1]  # Second positional arg is args
            assert args.include_raw is True


def test_job_replay_command(tmp_path):
    """Test that job replay command works with a valid debug file."""
    import json
    from typer.testing import CliRunner
    from metriq_gym.cli import app

    runner = CliRunner()

    # Create a debug file for QML Kernel benchmark
    debug_data = {
        "job_id": "test-replay-job",
        "job_type": "QML Kernel",
        "params": {
            "benchmark_name": "QML Kernel",
            "num_qubits": 4,
            "shots": 10,
        },
        "job_data": {
            "provider_job_ids": ["prov-123"],
        },
        "raw_results": [
            {
                "measurement_counts": {"0000": 10},
                "shots": 10,
                "num_measured_qubits": 4,
            }
        ],
    }

    debug_file = tmp_path / "debug.json"
    with open(debug_file, "w") as f:
        json.dump(debug_data, f)

    result = runner.invoke(app, ["job", "replay", str(debug_file)])

    assert result.exit_code == 0
    assert "Replaying QML Kernel benchmark" in result.stdout
    assert "Replay completed successfully" in result.stdout
    assert "accuracy_score" in result.stdout


def test_job_replay_command_with_json_output(tmp_path):
    """Test that job replay command can output to JSON file."""
    import json
    from typer.testing import CliRunner
    from metriq_gym.cli import app

    runner = CliRunner()

    # Create a debug file
    debug_data = {
        "job_id": "test-replay-job",
        "job_type": "QML Kernel",
        "params": {
            "benchmark_name": "QML Kernel",
            "num_qubits": 4,
            "shots": 10,
        },
        "job_data": {
            "provider_job_ids": ["prov-123"],
        },
        "raw_results": [
            {
                "measurement_counts": {"0000": 10},
                "shots": 10,
                "num_measured_qubits": 4,
            }
        ],
    }

    debug_file = tmp_path / "debug.json"
    output_file = tmp_path / "output.json"
    with open(debug_file, "w") as f:
        json.dump(debug_data, f)

    result = runner.invoke(app, ["job", "replay", str(debug_file), "--json", str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file) as f:
        output_data = json.load(f)
    assert "accuracy_score" in output_data


def test_job_replay_command_missing_file():
    """Test that job replay command fails gracefully with missing file."""
    from typer.testing import CliRunner
    from metriq_gym.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["job", "replay", "/nonexistent/file.json"])

    assert result.exit_code == 1
    assert "Debug file not found" in result.stdout
