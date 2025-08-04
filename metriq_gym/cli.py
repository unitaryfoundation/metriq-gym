"""Command-line parsing for running metriq benchmarks."""

import argparse
import logging

from tabulate import tabulate

from qbraid.runtime import get_providers
from metriq_gym.job_manager import JobManager, MetriqGymJob


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LIST_JOBS_HEADERS = ["Metriq-gym Job Id", "Provider", "Device", "Type", "Dispatch time (UTC)"]
LIST_JOBS_HEADERS_FULL = ["Suite ID"] + LIST_JOBS_HEADERS

LATEST_JOB_ID = "latest"

JOB_ID_ARGUMENT_NAME = "job_id"


def list_jobs(
    jobs: list[MetriqGymJob], show_index: bool = False, show_suite_id: bool = True
) -> None:
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
            [job.to_table_row(show_suite_id=show_suite_id) for job in jobs],
            headers=LIST_JOBS_HEADERS if not show_suite_id else LIST_JOBS_HEADERS_FULL,
            tablefmt="grid",
            showindex=show_index,
        )
    )


def prompt_for_job(args: argparse.Namespace, job_manager: JobManager) -> MetriqGymJob | None:
    if args.job_id:
        if args.job_id == LATEST_JOB_ID:
            return job_manager.get_latest_job()
        return job_manager.get_job(args.job_id)
    jobs = job_manager.get_jobs()
    if not jobs:
        print("No jobs found.")
        return None
    print("Available jobs:")
    list_jobs(jobs, show_index=True)
    selected_index: int
    user_input: str
    while True:
        try:
            user_input = input("Select a job index (or 'q' for quit): ")
            if user_input.lower() == "q":
                print("\nExiting...")
                return None
            selected_index = int(user_input)
            if 0 <= selected_index < len(jobs):
                break
            else:
                print(f"Invalid index. Please enter a number between 0 and {len(jobs) - 1}")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    return jobs[selected_index]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the metriq-gym benchmarking CLI.

    This function sets up the complete argument parsing structure for metriq-gym,
    supporting dispatch of multiple benchmark configuration files for comprehensive
    device characterization.

    Returns:
        Parsed arguments as an argparse.Namespace object containing all CLI options.

    Dispatch mode:
        - Multiple benchmarks: Requires one or more JSON configuration files
        - Each file can contain different parameters for any benchmark type
        - Same benchmark type can be run multiple times with different configurations
    """
    parser = argparse.ArgumentParser(description="Metriq-Gym benchmarking CLI")
    subparsers = parser.add_subparsers(dest="action", required=True, help="Action to perform")

    dispatch_parser = subparsers.add_parser("dispatch", help="Dispatch jobs")
    dispatch_subparsers = dispatch_parser.add_subparsers(
        dest="dispatch_action", required=True, help="Action to perform"
    )
    job_dispatch_subparser = dispatch_subparsers.add_parser("job", help="Dispatch a single job")
    suite_dispatch_subparser = dispatch_subparsers.add_parser(
        "suite", help="Dispatch a suite of jobs"
    )

    job_dispatch_subparser.add_argument(
        "config",
        type=str,
        help="Path to job configuration file.",
    )

    suite_dispatch_subparser.add_argument(
        "suite_config",
        type=str,
        help="Path to suite configuration file.",
    )

    for subparser in [job_dispatch_subparser, suite_dispatch_subparser]:
        subparser.add_argument(
            "-p",
            "--provider",
            type=str,
            choices=get_providers() + ["local"],
            help="String identifier for backend provider service",
        )
        subparser.add_argument(
            "-d",
            "--device",
            type=str,
            help="Backend to use",
        )

    poll_parser = subparsers.add_parser("poll", help="Poll jobs")
    poll_subparsers = poll_parser.add_subparsers(
        dest="poll_action", required=True, help="Action to perform"
    )

    suite_poll_subparser = poll_subparsers.add_parser("suite", help="Get suite results")
    job_poll_subparser = poll_subparsers.add_parser("job", help="Get job results")

    job_poll_subparser.add_argument(
        JOB_ID_ARGUMENT_NAME, type=str, nargs="?", help="Job ID to poll (optional)"
    )
    suite_poll_subparser.add_argument(
        "suite_id",
        type=str,
        nargs="?",
        help="Suite ID to poll results for",
    )

    for subparser in [job_poll_subparser, suite_poll_subparser]:
        subparser.add_argument(
            "--json",
            nargs="?",
            required=False,
            default=argparse.SUPPRESS,
            help="Export results to JSON file (optional)",
        )

    view_parser = subparsers.add_parser("view", help="View jobs")
    view_subparsers = view_parser.add_subparsers(
        dest="view_action", required=True, help="Action to perform"
    )
    suite_view_subparser = view_subparsers.add_parser("suite", help="View suite jobs")
    job_view_subparser = view_subparsers.add_parser("job", help="View job details")
    job_view_subparser.add_argument(
        JOB_ID_ARGUMENT_NAME, type=str, nargs="?", help="Job ID to view (optional)"
    )
    suite_view_subparser.add_argument(
        "suite_id",
        type=str,
        nargs="?",
        help="Suite ID to view jobs for",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete jobs")
    delete_parser.add_argument(
        JOB_ID_ARGUMENT_NAME, type=str, nargs="?", help="Job ID to delete (optional)"
    )

    return parser.parse_args()
