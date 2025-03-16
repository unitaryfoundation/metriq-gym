"""Command-line parsing for running metriq benchmarks."""

import argparse
import sys

from tabulate import tabulate

from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.provider import ProviderType

LIST_JOBS_HEADERS = ["Metriq-gym Job Id", "Provider", "Device", "Type", "Dispatch time (UTC)"]


def list_jobs(jobs: list[MetriqGymJob], show_index: bool = False) -> None:
    """List jobs recorded in the job manager.

    Args:
        jobs: List of MetriqGymJob instances.
        show_index: Whether to show the job index in the output table.
    """
    if not jobs:
        print("No jobs found.")
        return
    # Display jobs in a tabular format.
    print(
        tabulate(
            [job.to_table_row() for job in jobs],
            headers=LIST_JOBS_HEADERS,
            tablefmt="grid",
            showindex=show_index,
        )
    )


def prompt_for_job(job_manager: JobManager) -> MetriqGymJob:
    jobs = job_manager.get_jobs()
    if not jobs:
        print("No jobs found.")
        sys.exit(0)

    print("Available jobs:")
    list_jobs(jobs, show_index=True)
    while True:
        try:
            selected_index = int(input("Select a job index: "))
            if 0 <= selected_index < len(jobs):
                break
            else:
                print(f"Invalid index. Please enter a number between 0 and {len(jobs) - 1}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    return jobs[selected_index]


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the quantum volume benchmark.

    Returns:
        Parsed arguments as an argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(description="Metriq-Gym benchmarking CLI")
    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform")

    dispatch_parser = subparsers.add_parser("dispatch", help="Dispatch jobs")
    dispatch_parser.add_argument(
        "input_file",
        type=str,
        help="Path to the file containing the benchmark parameters",
    )
    dispatch_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=ProviderType.value_list(),
        default="ibmq",
        help="String identifier for backend provider service",
    )
    dispatch_parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="qasm_simulator",
        help='Backend to use (default is "qasm_simulator")',
    )

    poll_parser = subparsers.add_parser("poll", help="Poll jobs")
    poll_parser.add_argument("--job_id", type=str, required=False, help="Job ID to poll (optional)")

    poll_parser = subparsers.add_parser("view", help="Poll jobs")
    poll_parser.add_argument("--job_id", type=str, required=False, help="Job ID to view (optional)")

    subparsers.add_parser("list-jobs", help="List dispatched jobs")

    return parser.parse_args()
