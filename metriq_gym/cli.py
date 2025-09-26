"""Command-line parsing for running metriq benchmarks."""

import argparse
import logging
import os

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
    resource_parsers = parser.add_subparsers(
        dest="resource", required=True, help="Resource (suite/job)"
    )

    # Suite resource group
    suite_parser = resource_parsers.add_parser("suite", help="Suite operations")
    suite_subparsers = suite_parser.add_subparsers(
        dest="action", required=True, help="Suite action"
    )

    suite_dispatch = suite_subparsers.add_parser("dispatch", help="Dispatch a suite of jobs")
    suite_dispatch.add_argument("suite_config", type=str, help="Path to suite configuration file.")
    suite_dispatch.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=get_providers() + ["local"],
        help="String identifier for backend provider service",
    )
    suite_dispatch.add_argument(
        "-d",
        "--device",
        type=str,
        help="Backend to use",
    )

    suite_poll = suite_subparsers.add_parser("poll", help="Poll suite jobs")
    suite_poll.add_argument(
        "suite_id",
        type=str,
        nargs="?",
        help="Suite ID to poll results for",
    )
    suite_poll.add_argument(
        "--json",
        nargs="?",
        required=False,
        default=argparse.SUPPRESS,
        help="Export results to JSON file (optional)",
    )

    suite_view = suite_subparsers.add_parser("view", help="View suite jobs")
    suite_view.add_argument(
        "suite_id",
        type=str,
        nargs="?",
        help="Suite ID to view jobs for",
    )

    suite_delete = suite_subparsers.add_parser("delete", help="Delete a suite")
    suite_delete.add_argument(
        "suite_id",
        type=str,
        nargs="?",
        help="Suite ID to delete",
    )

    # Job resource group
    job_parser = resource_parsers.add_parser("job", help="Job operations")
    job_subparsers = job_parser.add_subparsers(dest="action", required=True, help="Job action")

    job_dispatch = job_subparsers.add_parser("dispatch", help="Dispatch a single job")
    job_dispatch.add_argument("config", type=str, help="Path to job configuration file.")
    job_dispatch.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=get_providers() + ["local"],
        help="String identifier for backend provider service",
    )
    job_dispatch.add_argument(
        "-d",
        "--device",
        type=str,
        help="Backend to use",
    )

    job_poll = job_subparsers.add_parser("poll", help="Poll job")
    job_view = job_subparsers.add_parser("view", help="View job")
    job_delete = job_subparsers.add_parser("delete", help="Delete job")
    job_upload = job_subparsers.add_parser(
        "upload", help="Upload job result to a GitHub repo (opens a PR)"
    )

    for subparser in [job_poll, job_view, job_delete, job_upload]:
        subparser.add_argument(
            JOB_ID_ARGUMENT_NAME, type=str, nargs="?", help="Job ID to operate on (optional)"
        )

    job_poll.add_argument(
        "--json",
        nargs="?",
        required=False,
        default=argparse.SUPPRESS,
        help="Export results to JSON file (optional)",
    )

    job_upload.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("MGYM_UPLOAD_REPO", "unitaryfoundation/metriq-data"),
        help=(
            "Target GitHub repo in the form 'owner/repo' "
            "(env: MGYM_UPLOAD_REPO, default: unitaryfoundation/metriq-data)"
        ),
    )
    job_upload.add_argument(
        "--base",
        dest="base_branch",
        type=str,
        default=os.environ.get("MGYM_UPLOAD_BASE_BRANCH", "main"),
        help="Base branch for the PR (env: MGYM_UPLOAD_BASE_BRANCH, default: main)",
    )
    job_upload.add_argument(
        "--dir",
        dest="upload_dir",
        type=str,
        default=os.environ.get("MGYM_UPLOAD_DIR"),
        help=(
            "Directory in the repo to place the JSON file "
            "(env: MGYM_UPLOAD_DIR; default: metriq-gym/v<major.minor>/<provider>)"
        ),
    )
    job_upload.add_argument(
        "--branch",
        dest="branch_name",
        type=str,
        default=None,
        help="Branch name to create for the PR (default: mgym/upload-<job_id>)",
    )
    job_upload.add_argument(
        "--title",
        dest="pr_title",
        type=str,
        default=None,
        help="Pull request title (default: includes job id)",
    )
    job_upload.add_argument(
        "--body",
        dest="pr_body",
        type=str,
        default=None,
        help="Pull request body",
    )
    job_upload.add_argument(
        "--commit-message",
        dest="commit_message",
        type=str,
        default=None,
        help="Commit message (default: includes job id)",
    )
    job_upload.add_argument(
        "--clone-dir",
        dest="clone_dir",
        type=str,
        default=os.environ.get("MGYM_UPLOAD_CLONE_DIR"),
        help="Optional working dir to clone into (env: MGYM_UPLOAD_CLONE_DIR)",
    )

    suite_upload = suite_subparsers.add_parser(
        "upload", help="Upload suite results to GitHub (opens a PR or compare link)"
    )
    suite_upload.add_argument(
        "suite_id",
        type=str,
        nargs="?",
        help="Suite ID to upload",
    )
    suite_upload.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("MGYM_UPLOAD_REPO", "unitaryfoundation/metriq-data"),
        help=(
            "Target GitHub repo in the form 'owner/repo' "
            "(env: MGYM_UPLOAD_REPO, default: unitaryfoundation/metriq-data)"
        ),
    )
    suite_upload.add_argument(
        "--base",
        dest="base_branch",
        type=str,
        default=os.environ.get("MGYM_UPLOAD_BASE_BRANCH", "main"),
        help="Base branch for the PR (env: MGYM_UPLOAD_BASE_BRANCH, default: main)",
    )
    suite_upload.add_argument(
        "--dir",
        dest="upload_dir",
        type=str,
        default=os.environ.get("MGYM_UPLOAD_DIR"),
        help=(
            "Directory in the repo to place the JSON file "
            "(env: MGYM_UPLOAD_DIR; default: metriq-gym/v<major.minor>/<provider>)"
        ),
    )
    suite_upload.add_argument(
        "--branch",
        dest="branch_name",
        type=str,
        default=None,
        help="Branch name to create for the PR (default: mgym/upload-<job_id>)",
    )
    suite_upload.add_argument(
        "--title",
        dest="pr_title",
        type=str,
        default=None,
        help="Pull request title (optional)",
    )
    suite_upload.add_argument(
        "--body",
        dest="pr_body",
        type=str,
        default=None,
        help="Pull request body",
    )
    suite_upload.add_argument(
        "--commit-message",
        dest="commit_message",
        type=str,
        default=None,
        help="Commit message (optional)",
    )
    suite_upload.add_argument(
        "--clone-dir",
        dest="clone_dir",
        type=str,
        default=os.environ.get("MGYM_UPLOAD_CLONE_DIR"),
        help="Optional working dir to clone into (env: MGYM_UPLOAD_CLONE_DIR)",
    )

    return parser.parse_args()
