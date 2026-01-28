"""Command-line interface for running Metriq-Gym benchmarks using Typer.

Usage overview:
  - Dispatch a single job:
      mgym job dispatch path/to/config.json -p <provider> -d <device>
  - Poll latest job and write JSON results:
      mgym job poll latest --json results.json
  - Dispatch a suite of jobs:
      mgym suite dispatch path/to/suite.json -p <provider> -d <device>
  - Poll a suite:
      mgym suite poll <suite_id>
  - Dry-run upload (no network):
      mgym job upload latest --dry-run
"""

import argparse
import logging
from typing import Annotated, Optional

import typer

from tabulate import tabulate

from metriq_gym.job_manager import JobManager, MetriqGymJob


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LIST_JOBS_HEADERS = [
    "Metriq-gym Job Id",
    "Provider",
    "Device",
    "Type",
    "# Qubits",
    "Dispatch time (UTC)",
]
LIST_JOBS_HEADERS_FULL = ["Suite ID"] + LIST_JOBS_HEADERS

LATEST_JOB_ID = "latest"

# Main Typer app
app = typer.Typer(
    name="mgym",
    help="Metriq-Gym CLI — dispatch, poll, and upload results for quantum benchmarks",
    no_args_is_help=True,
)

# Sub-apps for job and suite commands
job_app = typer.Typer(help="Job operations", no_args_is_help=True)
suite_app = typer.Typer(help="Suite operations", no_args_is_help=True)

app.add_typer(job_app, name="job")
app.add_typer(suite_app, name="suite")


def list_jobs(
    jobs: list[MetriqGymJob], show_index: bool = False, show_suite_id: bool = True
) -> None:
    """List jobs recorded in the job manager.

    Args:
        jobs: List of MetriqGymJob instances.
        show_index: Whether to show the job index in the output table.
        show_suite_id: Whether to show the suite ID column.
    """
    if not jobs:
        print("No jobs found.")
        return
    print(
        tabulate(
            [job.to_table_row(show_suite_id=show_suite_id) for job in jobs],
            headers=LIST_JOBS_HEADERS if not show_suite_id else LIST_JOBS_HEADERS_FULL,
            tablefmt="grid",
            showindex=show_index,
        )
    )


def prompt_for_job(job_id: Optional[str], job_manager: JobManager) -> MetriqGymJob | None:
    """Prompt user to select a job if job_id is not provided.

    Args:
        job_id: Optional job ID or 'latest'.
        job_manager: JobManager instance.

    Returns:
        Selected MetriqGymJob or None.
    """
    if job_id:
        if job_id == LATEST_JOB_ID:
            return job_manager.get_latest_job()
        return job_manager.get_job(job_id)
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


# Type aliases for common options
ProviderOption = Annotated[
    Optional[str],
    typer.Option(
        "--provider",
        "-p",
        help="Provider name (e.g., ibm, braket, azure, ionq, local)",
    ),
]

DeviceOption = Annotated[
    Optional[str],
    typer.Option(
        "--device",
        "-d",
        help="Device identifier",
    ),
]


# -----------------------------------------------------------------------------
# Job commands
# -----------------------------------------------------------------------------


@job_app.command("dispatch")
def job_dispatch(
    config: Annotated[str, typer.Argument(help="Path to job configuration JSON file")],
    provider: ProviderOption = None,
    device: DeviceOption = None,
) -> None:
    """Dispatch a benchmark job to a quantum device or simulator."""
    from metriq_gym.run import dispatch_job as _dispatch_job

    args = argparse.Namespace()
    args.config = config
    args.provider = provider
    args.device = device

    job_manager = JobManager()
    _dispatch_job(args, job_manager)


@job_app.command("estimate")
def job_estimate(
    config: Annotated[str, typer.Argument(help="Path to job configuration JSON file")],
    provider: ProviderOption = None,
    device: DeviceOption = None,
) -> None:
    """Estimate circuit resource requirements before dispatching jobs.

    This is especially useful for understanding costs on paid hardware like Quantinuum.
    For Quantinuum providers, calculates H-series Quantum Credits (HQCs).
    """
    from metriq_gym.run import estimate_job as _estimate_job

    args = argparse.Namespace()
    args.config = config
    args.provider = provider
    args.device = device

    _estimate_job(args)


@job_app.command("poll")
def job_poll(
    job_id: Annotated[
        Optional[str], typer.Argument(help="Job ID to poll (use 'latest' for most recent)")
    ] = None,
    json_output: Annotated[
        Optional[str],
        typer.Option("--json", help="Export results to JSON file"),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Ignore locally cached results and refetch"),
    ] = False,
    include_raw: Annotated[
        bool,
        typer.Option(
            "--include-raw",
            help="Include raw measurement counts in JSON export for debugging/replay",
        ),
    ] = False,
) -> None:
    """Poll job status and retrieve results when complete."""
    from metriq_gym.run import poll_job as _poll_job

    args = argparse.Namespace()
    args.job_id = job_id
    args.no_cache = no_cache
    args.include_raw = include_raw
    if json_output is not None:
        args.json = json_output

    job_manager = JobManager()
    _poll_job(args, job_manager)


@job_app.command("view")
def job_view(
    job_id: Annotated[
        Optional[str], typer.Argument(help="Job ID to view (lists all if omitted)")
    ] = None,
) -> None:
    """View job details and metadata."""
    from metriq_gym.run import view_job as _view_job

    args = argparse.Namespace()
    args.job_id = job_id

    job_manager = JobManager()
    _view_job(args, job_manager)


@job_app.command("delete")
def job_delete(
    job_id: Annotated[Optional[str], typer.Argument(help="Job ID to delete")] = None,
) -> None:
    """Delete a job from the local database.

    Note: This only removes the job from local tracking. It does not cancel
    jobs running on quantum hardware.
    """
    from metriq_gym.run import delete_job as _delete_job

    args = argparse.Namespace()
    args.job_id = job_id

    job_manager = JobManager()
    _delete_job(args, job_manager)


@job_app.command("upload")
def job_upload(
    job_id: Annotated[Optional[str], typer.Argument(help="Job ID to upload")] = None,
    repo: Annotated[
        str,
        typer.Option(
            "--repo",
            help="Target GitHub repo (owner/repo)",
            envvar="MGYM_UPLOAD_REPO",
        ),
    ] = "unitaryfoundation/metriq-data",
    base_branch: Annotated[
        str,
        typer.Option(
            "--base",
            help="Base branch for the PR",
            envvar="MGYM_UPLOAD_BASE_BRANCH",
        ),
    ] = "main",
    upload_dir: Annotated[
        Optional[str],
        typer.Option(
            "--dir",
            help="Directory in repo for the JSON file",
            envvar="MGYM_UPLOAD_DIR",
        ),
    ] = None,
    branch_name: Annotated[
        Optional[str],
        typer.Option("--branch", help="Branch name for the PR"),
    ] = None,
    pr_title: Annotated[
        Optional[str],
        typer.Option("--title", help="Pull request title"),
    ] = None,
    pr_body: Annotated[
        Optional[str],
        typer.Option("--body", help="Pull request body"),
    ] = None,
    commit_message: Annotated[
        Optional[str],
        typer.Option("--commit-message", help="Commit message"),
    ] = None,
    clone_dir: Annotated[
        Optional[str],
        typer.Option(
            "--clone-dir",
            help="Working directory to clone into",
            envvar="MGYM_UPLOAD_CLONE_DIR",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Do not push or open a PR; print actions only"),
    ] = False,
) -> None:
    """Upload job results to GitHub via pull request."""
    from metriq_gym.run import upload_job as _upload_job

    args = argparse.Namespace()
    args.job_id = job_id
    args.repo = repo
    args.base_branch = base_branch
    args.upload_dir = upload_dir
    args.branch_name = branch_name
    args.pr_title = pr_title
    args.pr_body = pr_body
    args.commit_message = commit_message
    args.clone_dir = clone_dir
    args.dry_run = dry_run

    job_manager = JobManager()
    _upload_job(args, job_manager)


@job_app.command("replay")
def job_replay(
    debug_file: Annotated[
        str,
        typer.Argument(help="Path to debug JSON file (created with --include-raw)"),
    ],
    json_output: Annotated[
        Optional[str],
        typer.Option("--json", help="Export replayed results to JSON file"),
    ] = None,
) -> None:
    """Replay benchmark computation from a debug file.

    This allows recomputing benchmark results locally without access to the
    original quantum provider, using the raw measurement data captured with
    --include-raw.
    """
    from metriq_gym.run import replay_from_debug_file

    result = replay_from_debug_file(debug_file)
    if result is None:
        raise typer.Exit(1)

    if json_output:
        import json

        with open(json_output, "w") as f:
            json.dump(result.model_dump(), f, indent=4)
        print(f"Replayed results exported to {json_output}")
    else:
        # Print results to CLI
        print("\n=== Replayed Results ===")
        for key, value in result.model_dump().items():
            print(f"{key}: {value}")


# -----------------------------------------------------------------------------
# Suite commands
# -----------------------------------------------------------------------------


@suite_app.command("dispatch")
def suite_dispatch(
    suite_config: Annotated[str, typer.Argument(help="Path to suite configuration file")],
    provider: ProviderOption = None,
    device: DeviceOption = None,
) -> None:
    """Dispatch a suite of benchmark jobs to a quantum device."""
    from metriq_gym.run import dispatch_suite as _dispatch_suite

    args = argparse.Namespace()
    args.suite_config = suite_config
    args.provider = provider
    args.device = device

    job_manager = JobManager()
    _dispatch_suite(args, job_manager)


@suite_app.command("poll")
def suite_poll(
    suite_id: Annotated[Optional[str], typer.Argument(help="Suite ID to poll")] = None,
    json_output: Annotated[
        Optional[str],
        typer.Option("--json", help="Export results to JSON file"),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Ignore locally cached results and refetch"),
    ] = False,
) -> None:
    """Poll suite jobs and retrieve results when complete."""
    from metriq_gym.run import poll_suite as _poll_suite

    args = argparse.Namespace()
    args.suite_id = suite_id
    args.no_cache = no_cache
    if json_output is not None:
        args.json = json_output

    job_manager = JobManager()
    _poll_suite(args, job_manager)


@suite_app.command("view")
def suite_view(
    suite_id: Annotated[Optional[str], typer.Argument(help="Suite ID to view")] = None,
) -> None:
    """View jobs in a suite."""
    from metriq_gym.run import view_suite as _view_suite

    args = argparse.Namespace()
    args.suite_id = suite_id

    job_manager = JobManager()
    _view_suite(args, job_manager)


@suite_app.command("delete")
def suite_delete(
    suite_id: Annotated[Optional[str], typer.Argument(help="Suite ID to delete")] = None,
) -> None:
    """Delete all jobs in a suite from the local database."""
    from metriq_gym.run import delete_suite as _delete_suite

    args = argparse.Namespace()
    args.suite_id = suite_id

    job_manager = JobManager()
    _delete_suite(args, job_manager)


@suite_app.command("upload")
def suite_upload(
    suite_id: Annotated[Optional[str], typer.Argument(help="Suite ID to upload")] = None,
    repo: Annotated[
        str,
        typer.Option(
            "--repo",
            help="Target GitHub repo (owner/repo)",
            envvar="MGYM_UPLOAD_REPO",
        ),
    ] = "unitaryfoundation/metriq-data",
    base_branch: Annotated[
        str,
        typer.Option(
            "--base",
            help="Base branch for the PR",
            envvar="MGYM_UPLOAD_BASE_BRANCH",
        ),
    ] = "main",
    upload_dir: Annotated[
        Optional[str],
        typer.Option(
            "--dir",
            help="Directory in repo for the JSON file",
            envvar="MGYM_UPLOAD_DIR",
        ),
    ] = None,
    branch_name: Annotated[
        Optional[str],
        typer.Option("--branch", help="Branch name for the PR"),
    ] = None,
    pr_title: Annotated[
        Optional[str],
        typer.Option("--title", help="Pull request title"),
    ] = None,
    pr_body: Annotated[
        Optional[str],
        typer.Option("--body", help="Pull request body"),
    ] = None,
    commit_message: Annotated[
        Optional[str],
        typer.Option("--commit-message", help="Commit message"),
    ] = None,
    clone_dir: Annotated[
        Optional[str],
        typer.Option(
            "--clone-dir",
            help="Working directory to clone into",
            envvar="MGYM_UPLOAD_CLONE_DIR",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Do not push or open a PR; print actions only"),
    ] = False,
) -> None:
    """Upload suite results to GitHub via pull request."""
    from metriq_gym.run import upload_suite as _upload_suite

    args = argparse.Namespace()
    args.suite_id = suite_id
    args.repo = repo
    args.base_branch = base_branch
    args.upload_dir = upload_dir
    args.branch_name = branch_name
    args.pr_title = pr_title
    args.pr_body = pr_body
    args.commit_message = commit_message
    args.clone_dir = clone_dir
    args.dry_run = dry_run

    job_manager = JobManager()
    _upload_suite(args, job_manager)
