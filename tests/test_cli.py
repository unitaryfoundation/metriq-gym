from datetime import datetime
from tabulate import tabulate
import pytest
from unittest.mock import MagicMock
from metriq_gym.cli import LIST_JOBS_HEADERS, list_jobs
from metriq_gym.benchmarks import JobType
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
            provider_name="ibmq",
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

    list_jobs(mock_jobs, show_index=False)

    # Capture the output
    captured = capsys.readouterr()

    # Expected output using tabulate
    table = [
        ["1234", "ibmq", "ibmq_qasm_simulator", "Quantum Volume", "2021-09-01T12:00:00"],
        ["5678", "ionq", "ionq_simulator", "Quantum Volume", "2021-09-02T12:00:00"],
    ]
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
