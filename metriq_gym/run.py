"""Runtime entrypoints for dispatching and managing metriq-gym benchmarks via the CLI."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import os
import sys
import logging
import uuid
from dotenv import load_dotenv

from tabulate import tabulate
from typing import Any, TYPE_CHECKING, Optional
from metriq_gym import __version__
from metriq_gym.cli import list_jobs, prompt_for_job, app as typer_app
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.qplatform.job import total_execution_time
from metriq_gym.schema_validator import load_and_validate, validate_and_create_model
from metriq_gym.constants import JobType
from metriq_gym.resource_estimation import (
    CircuitBatch,
    aggregate_resource_estimates,
    print_resource_estimate,
    quantinuum_hqc_formula,
)
from metriq_gym.suite_parser import parse_suite_file
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.upload_paths import default_upload_dir, job_filename, suite_filename


if TYPE_CHECKING:
    from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("metriq_gym")

DEFAULT_UPLOAD_PR_LABELS = ["data", "source:metriq-gym"]


@dataclass
class FetchResultOutput:
    """Container for fetch_result output, including optional raw data for debugging."""

    result: "BenchmarkResult"
    raw_counts: list[dict[str, Any]] | None = None
    from_cache: bool = False


def load_provider(provider_name: str):
    """Lazy proxy to qbraid.runtime.load_provider.

    Exposed at module level so tests can monkeypatch `metriq_gym.run.load_provider`.
    """
    from qbraid.runtime import load_provider as _load_provider

    return _load_provider(provider_name)


def get_providers() -> list[str]:
    """Lazy proxy to qbraid.runtime.get_providers.

    Exposed at module level so tests can monkeypatch `metriq_gym.run.get_providers`.
    """
    from qbraid.runtime import get_providers as _get_providers

    return _get_providers()


def load_job(job_id: str, *, provider: str, **kwargs):
    """Lazy proxy to qbraid.runtime.load_job.

    Exposed at module level so tests can monkeypatch `metriq_gym.run.load_job`.
    """
    from qbraid.runtime import load_job as _load_job

    return _load_job(job_id, provider=provider, **kwargs)


def job_status(quantum_job):
    """Lazy proxy to metriq_gym.qplatform.job.job_status.

    Exposed so tests and callers can patch `metriq_gym.run.job_status` and to
    keep heavy imports out of CLI cold start.
    """
    from metriq_gym.qplatform.job import job_status as _job_status

    return _job_status(quantum_job)


def _lazy_registry():
    # Import registry lazily to avoid importing heavy benchmark modules on CLI startup
    from . import registry as _registry

    return _registry


COMMON_SUITE_METADATA = {
    "provider": ("platform", "provider"),
    "device": ("platform", "device"),
    "timestamp": ("timestamp",),
    "app_version": ("app_version",),
}


def setup_device(provider_name: str, device_name: str):
    """
    Setup a QBraid device with id device_name from specified provider.

    Args:
        provider_name: a metriq-gym supported provider name.
        device_name: the id of a device supported by the provider.
    Raises:
        QBraidSetupError: If no device matching the name is found in the provider.
    """
    from qbraid import QbraidError
    from metriq_gym.exceptions import QBraidSetupError

    if not provider_name:
        providers = ", ".join(get_providers())
        logger.error("No provider name specified.")
        logger.error(f"Providers available: {providers}")
        raise QBraidSetupError("Provider not found")

    try:
        provider = load_provider(provider_name)
    except QbraidError:
        providers = ", ".join(get_providers())
        logger.error(f"No provider matching the name '{provider_name}' found.")
        logger.error(f"Providers available: {providers}")
        raise QBraidSetupError("Provider not found")

    if not device_name:
        devices = ", ".join([device.id for device in provider.get_devices()])
        logger.error("No device name specified.")
        logger.error(f"Devices available: {devices}")
        raise QBraidSetupError("Device not found")

    try:
        device = provider.get_device(device_name)
    except QbraidError:
        devices = ", ".join([device.id for device in provider.get_devices()])
        logger.error(
            f"No device matching the name '{device_name}' found in provider '{provider_name}'."
        )
        logger.error(f"Devices available: {devices}")
        raise QBraidSetupError("Device not found")
    return device


def setup_benchmark(args, params, job_type: JobType) -> "Benchmark":
    reg = _lazy_registry()
    return reg.BENCHMARK_HANDLERS[job_type](args, params)


def setup_job_data_class(job_type: JobType) -> type["BenchmarkData"]:
    reg = _lazy_registry()
    return reg.BENCHMARK_DATA_CLASSES[job_type]


def setup_benchmark_result_class(job_type: JobType) -> type["BenchmarkResult"]:
    reg = _lazy_registry()
    return reg.BENCHMARK_RESULT_CLASSES[job_type]


def dispatch_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch a single benchmark configuration to a quantum device.

    Args:
        args: CLI arguments with benchmark config, provider, and device
        job_manager: Tracks dispatched jobs for later polling

    Note: Continues processing remaining configs if individual configs fail.
    """
    print(f"Starting dispatch on {args.provider}:{args.device}...")

    try:
        device = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    config_file = args.config

    if not os.path.exists(config_file):
        print(f"✗ {config_file}: Configuration file not found")
        return

    # Load and validate the benchmark configuration
    params = load_and_validate(config_file)

    # Validate that the benchmark exists
    reg = _lazy_registry()
    if params.benchmark_name not in reg.get_available_benchmarks():
        print(
            f"✗ {config_file}: Unsupported benchmark '{params.benchmark_name}'. Available: {reg.get_available_benchmarks()}"
        )
        return

    job_type = JobType(params.benchmark_name)

    print(f"Dispatching {params.benchmark_name}...")

    handler: Benchmark = setup_benchmark(args, params, job_type)
    job_data: BenchmarkData = handler.dispatch_handler(device)

    # Lazy import to avoid heavy modules during CLI cold start
    from metriq_gym.qplatform.device import normalized_metadata

    job_id = job_manager.add_job(
        MetriqGymJob(
            id=str(uuid.uuid4()),
            job_type=job_type,
            params=params.model_dump(exclude_none=True),
            data=asdict(job_data),
            provider_name=args.provider,
            device_name=args.device,
            platform={
                "provider": args.provider,
                "device": args.device,
                "device_metadata": normalized_metadata(device),
            },
            dispatch_time=datetime.now(),
        )
    )

    print(f"✓ {params.benchmark_name} dispatched with metriq-gym Job ID: {job_id}")


def dispatch_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch multiple benchmark configurations to a quantum device.

    Enables comprehensive device characterization by running the same benchmark
    type with different parameters or multiple benchmark types in sequence.

    Args:
        args: CLI arguments with benchmark configs, provider, and device
        job_manager: Tracks dispatched jobs for later polling

    Note: Continues processing remaining configs if individual configs fail.
    """
    print(f"Starting suite dispatch on {args.provider}:{args.device}...")

    try:
        device = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    config_file = args.suite_config

    if not os.path.exists(config_file):
        print(f"✗ {config_file}: Configuration file not found")
        return

    # Load and validate the benchmark configuration
    suite = parse_suite_file(config_file)
    if not suite.benchmarks:
        print(f"✗ {config_file}: No benchmarks found in the suite")
        return

    # Lazy import once per function call
    from metriq_gym.qplatform.device import normalized_metadata

    results = []
    successful_jobs = []

    suite_id = str(uuid.uuid4())
    for benchmark_entry in suite.benchmarks:
        try:
            params = validate_and_create_model(benchmark_entry.config)

            # Validate that the benchmark exists
            reg = _lazy_registry()
            if params.benchmark_name not in reg.get_available_benchmarks():
                results.append(
                    f"✗ {config_file}: Unsupported benchmark '{params.benchmark_name}'. Available: {reg.get_available_benchmarks()}"
                )
                continue

            job_type = JobType(params.benchmark_name)

            print(
                f"Dispatching {benchmark_entry.name} ({params.benchmark_name}) from {suite.name}..."
            )

            handler: Benchmark = setup_benchmark(args, params, job_type)
            job_data: BenchmarkData = handler.dispatch_handler(device)

            job_id = job_manager.add_job(
                MetriqGymJob(
                    suite_id=suite_id,
                    suite_name=suite.name,
                    id=str(uuid.uuid4()),
                    job_type=job_type,
                    params=params.model_dump(exclude_none=True),
                    data=asdict(job_data),
                    provider_name=args.provider,
                    device_name=args.device,
                    platform={
                        "provider": args.provider,
                        "device": args.device,
                        "device_metadata": normalized_metadata(device),
                    },
                    dispatch_time=datetime.now(),
                )
            )

            results.append(
                f"✓ {benchmark_entry.name} ({params.benchmark_name}) from {suite.name} dispatched with metriq-gym Job ID: {job_id}"
            )
            successful_jobs.append(job_id)

        except Exception as e:
            error_details = f"{type(e).__name__}: {str(e)}"
            results.append(f"✗ {benchmark_entry.name} from {suite.name} failed: {error_details}")

    print("\nSummary:")
    for result in results:
        print(f"  {result}")

    print(f"\nDispatch complete for suite {suite.name} with metriq-gym Suite ID {suite_id}")
    print(f"\nSuccessfully dispatched {len(successful_jobs)}/{len(suite.benchmarks)} benchmarks.")
    if successful_jobs:
        print("Use 'mgym suite poll' or 'mgym job poll' to check suite/job status.")


def poll_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args.job_id, job_manager)
    if not metriq_job:
        return
    print("Polling job...")
    fetch_output = fetch_result(metriq_job, args, job_manager)
    if fetch_output is None:
        print(f"Job {metriq_job.id} is not yet completed or has no results.")
        return
    export_job_result(
        args,
        metriq_job,
        fetch_output.result,
        raw_counts=fetch_output.raw_counts,
        from_cache=fetch_output.from_cache,
    )


def upload_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Upload a job's results to a GitHub repo by opening a Pull Request."""
    metriq_job = prompt_for_job(args.job_id, job_manager)
    if not metriq_job:
        return
    print("Preparing job upload...")
    fetch_output = fetch_result(metriq_job, args, job_manager)
    if fetch_output is None:
        print(f"Job {metriq_job.id} is not yet completed or has no results.")
        return
    result = fetch_output.result

    repo = getattr(args, "repo", None)
    if not repo:
        print("Error: --repo not provided and MGYM_UPLOAD_REPO not set.")
        return

    base_branch = getattr(args, "base_branch", "main")
    provider = metriq_job.provider_name
    device = metriq_job.device_name
    # Default upload dir: <root>/v<major.minor>/<provider>/<device>
    upload_dir = getattr(args, "upload_dir", None) or default_upload_dir(
        __version__, provider, device
    )
    branch_name = getattr(args, "branch_name", None)
    pr_title = getattr(args, "pr_title", None)
    pr_body = getattr(args, "pr_body", None)
    commit_message = getattr(args, "commit_message", None)
    clone_dir = getattr(args, "clone_dir", None)
    dry_run = getattr(args, "dry_run", False)

    # Write this job's record to a dedicated JSON file in the target directory
    from metriq_gym.exporters.dict_exporter import DictExporter

    record = DictExporter(metriq_job, result).export() | {"params": metriq_job.params}

    try:
        from metriq_gym.exporters.github_pr_exporter import GitHubPRExporter

        url = GitHubPRExporter(metriq_job, result).export(
            repo=repo,
            base_branch=base_branch,
            directory=upload_dir,
            branch_name=branch_name,
            commit_message=commit_message,
            pr_title=pr_title,
            pr_body=pr_body,
            pr_labels=DEFAULT_UPLOAD_PR_LABELS,
            clone_dir=clone_dir,
            payload=record,
            filename=job_filename(metriq_job, payload=record),
            append=False,
            dry_run=dry_run,
        )
        if url.startswith("DRY-RUN:"):
            print(url)
        elif "/compare/" in url:
            print("✓ Branch pushed to your fork.")
            print(f"Open this URL to create the PR: {url}")
        else:
            print(f"✓ Opened pull request: {url}")
    except Exception as e:
        print(f"✗ Upload failed: {e}")


def poll_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    if not args.suite_id:
        print("Suite ID is required to poll suite results.")
        return
    jobs = job_manager.get_jobs_by_suite_id(args.suite_id)
    if not jobs:
        print(f"No jobs found for suite ID {args.suite_id}.")
        return
    results: list[Any] = []
    for metriq_job in jobs:
        fetch_output = fetch_result(metriq_job, args, job_manager)
        if fetch_output is None:
            print(f"Job {metriq_job.id} is not yet completed or has no results.")
            return
        results.append(fetch_output.result)
    export_suite_results(args, jobs, results)


def _get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    current: Any = mapping
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def print_selected(d, selected_keys):
    for label, path in selected_keys.items():
        value = _get_nested(d, path)
        if value is not None:
            print(f"{label}: {value}")


def export_suite_results(args, jobs: list[MetriqGymJob], results: list["BenchmarkResult"]) -> None:
    if not jobs:
        return

    from metriq_gym.exporters.dict_exporter import DictExporter

    records = []
    for job, result in zip(jobs, results):
        records.append(DictExporter(job, result).export() | {"params": job.params})

    if hasattr(args, "json"):
        raise NotImplementedError("JSON export of suite results is not implemented yet.")
    else:
        print("\n--- Suite Metadata ---")
        print_selected(records[0], COMMON_SUITE_METADATA)
        print("\n--- Suite Results ---")
        print(tabulate_job_results(records))


def upload_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Upload all jobs in a suite as a single JSON (array of job records) in one PR."""
    if not args.suite_id:
        print("Suite ID is required to upload suite results.")
        return
    jobs = job_manager.get_jobs_by_suite_id(args.suite_id)
    if not jobs:
        print(f"No jobs found for suite ID {args.suite_id}.")
        return

    repo = getattr(args, "repo", None)
    if not repo:
        print("Error: --repo not provided and MGYM_UPLOAD_REPO not set.")
        return

    # Ensure all results are available first
    results: list[Any] = []
    for metriq_job in jobs:
        fetch_output = fetch_result(metriq_job, args, job_manager)
        if fetch_output is None:
            print(f"Job {metriq_job.id} is not yet completed or has no results.")
            return
        results.append(fetch_output.result)

    # Build array of per-job records (no common header)
    from metriq_gym.exporters.dict_exporter import DictExporter

    records: list[dict] = []
    for job, result in zip(jobs, results):
        records.append(DictExporter(job, result).export() | {"params": job.params})

    # Use provider/device and suite name from first job for PR title only
    provider = jobs[0].provider_name
    device = jobs[0].device_name
    suite_name = jobs[0].suite_name

    base_branch = getattr(args, "base_branch", "main")
    upload_dir = getattr(args, "upload_dir", None) or default_upload_dir(
        __version__, provider, device
    )
    branch_name = getattr(args, "branch_name", None) or f"mgym/upload-suite-{args.suite_id}"
    # Prefer suite name; avoid falling back to suite_id in the title
    suite_label = suite_name or "unnamed"
    pr_title = getattr(args, "pr_title", None) or (
        f"mgym upload: suite {suite_label} on {provider}/{device}"
    )
    pr_body = getattr(args, "pr_body", None)
    # Default commit message aligns with PR title to make browser compare pre-fill useful
    commit_message = getattr(args, "commit_message", None) or pr_title
    clone_dir = getattr(args, "clone_dir", None)
    dry_run = getattr(args, "dry_run", False)

    try:
        from metriq_gym.exporters.github_pr_exporter import GitHubPRExporter

        url = GitHubPRExporter(jobs[0], results[0]).export(
            repo=repo,
            base_branch=base_branch,
            directory=upload_dir,
            branch_name=branch_name,
            commit_message=commit_message,
            pr_title=pr_title,
            pr_body=pr_body,
            pr_labels=DEFAULT_UPLOAD_PR_LABELS,
            clone_dir=clone_dir,
            payload=records,
            filename=suite_filename(suite_name, jobs[0].dispatch_time, payload=records),
            append=False,
            dry_run=dry_run,
        )
        if url.startswith("DRY-RUN:"):
            print(url)
        elif "/compare/" in url:
            print("✓ Branch pushed to your fork.")
            print(f"Open this URL to create the PR: {url}")
        else:
            print(f"✓ Opened pull request: {url}")
    except Exception as e:
        print(f"✗ Upload failed: {e}")


def tabulate_job_results(records, sep=" +/- "):
    rows = []
    metric_keys = set()
    for record in records:
        metric_keys.update(record.get("results", {}).get("values", {}).keys())
    metric_keys = sorted(metric_keys)

    headers = ["Job Type", "Parameters"] + metric_keys

    for record in records:
        name = record.get("job_type")
        params = record.get("params", {})
        if isinstance(params, dict):
            params_str = ", ".join(
                f"{k}={v}" for k, v in sorted(params.items()) if k != "benchmark_name"
            )
        else:
            params_str = str(params)
        row = [name, params_str]
        values = record.get("results", {}).get("values", {})
        uncertainties = record.get("results", {}).get("uncertainties", {})
        for metric in metric_keys:
            value = values.get(metric, "")
            uncertainty = uncertainties.get(metric)
            if uncertainty is None or uncertainty == "":
                row.append(value)
            else:
                row.append(f"{value}{sep}{uncertainty}")
        rows.append(row)

    return tabulate(rows, headers=headers, floatfmt=".4g")


def export_job_result(
    args: argparse.Namespace,
    metriq_job: MetriqGymJob,
    result: "BenchmarkResult",
    raw_counts: list[dict[str, Any]] | None = None,
    from_cache: bool = False,
) -> None:
    """Export job results to JSON or CLI.

    Args:
        args: CLI arguments.
        metriq_job: The job being exported.
        result: The benchmark result.
        raw_counts: Optional raw measurement counts for debugging.
        from_cache: Whether result was loaded from cache (raw counts unavailable).
    """
    include_raw = getattr(args, "include_raw", False)
    if include_raw and from_cache:
        print(
            "Warning: --include-raw requested but results are from cache. "
            "Raw counts not available. Use --no-cache to refetch from provider."
        )

    if hasattr(args, "json"):
        from metriq_gym.exporters.json_exporter import JsonExporter

        JsonExporter(metriq_job, result).export(args.json)

        # Write raw debug data to separate file if requested
        if include_raw and raw_counts is not None:
            _export_raw_debug_data(args.json, metriq_job, raw_counts)
    else:
        from metriq_gym.exporters.cli_exporter import CliExporter

        CliExporter(metriq_job, result).export()


def _export_raw_debug_data(
    base_filename: str,
    metriq_job: MetriqGymJob,
    raw_counts: list[dict[str, Any]],
) -> None:
    """Export raw measurement counts and job data to a separate debug file.

    Args:
        base_filename: The base JSON filename (e.g., 'result.json').
        metriq_job: The job being exported.
        raw_counts: Raw measurement counts from the provider.
    """
    import json

    # Generate debug filename: result.json -> result_debug.json
    if base_filename.endswith(".json"):
        debug_filename = base_filename[:-5] + "_debug.json"
    else:
        debug_filename = base_filename + "_debug.json"

    debug_data = {
        "job_id": metriq_job.id,
        "job_type": metriq_job.job_type.value,
        "params": metriq_job.params,
        "job_data": metriq_job.data,
        "raw_counts": raw_counts,
    }

    with open(debug_filename, "w") as f:
        json.dump(debug_data, f, indent=4)
    print(f"Debug data exported to {debug_filename}")


def fetch_result(
    metriq_job: MetriqGymJob, args: argparse.Namespace, job_manager: JobManager
) -> Optional[FetchResultOutput]:
    """Fetch benchmark results, optionally including raw measurement counts.

    Args:
        metriq_job: The job to fetch results for.
        args: CLI arguments, may include 'include_raw' flag.
        job_manager: Job manager for persisting results.

    Returns:
        FetchResultOutput containing the result, optional raw counts, and cache status.
        Returns None if job is not yet completed.
    """
    include_raw = getattr(args, "include_raw", False)
    job_type: JobType = JobType(metriq_job.job_type)
    job_result_type = setup_benchmark_result_class(job_type)
    if metriq_job.result_data is not None and not getattr(args, "no_cache", False):
        print("[Cached result data]")
        cached_result = job_result_type.model_validate(metriq_job.result_data)
        return FetchResultOutput(result=cached_result, raw_counts=None, from_cache=True)

    job_data: "BenchmarkData" = setup_job_data_class(job_type)(**metriq_job.data)
    handler: Benchmark = setup_benchmark(
        args, validate_and_create_model(metriq_job.params), job_type
    )
    from qbraid.runtime import JobStatus

    quantum_jobs = [
        (load_job(job_id, provider=metriq_job.provider_name, **asdict(job_data)))
        for job_id in job_data.provider_job_ids
    ]
    if all(task.status() == JobStatus.COMPLETED for task in quantum_jobs):
        result_data = [task.result().data for task in quantum_jobs]

        # Serialize raw counts if requested
        raw_counts = None
        if include_raw:
            from metriq_gym.helpers.task_helpers import serialize_raw_counts

            raw_counts = serialize_raw_counts(result_data)

        # Compute total execution time across all jobs, if provided by the backend
        total_time = total_execution_time(quantum_jobs)
        if total_time is not None:
            print(f"Total execution time across {len(quantum_jobs)} jobs: {total_time:.2f} seconds")
        else:
            logger.debug("Failed to compute benchmark runtime.", exc_info=True)
        metriq_job.runtime_seconds = total_time

        result: "BenchmarkResult" = handler.poll_handler(job_data, result_data, quantum_jobs)
        # Cache result_data in metriq_job, excluding computed fields like 'score'
        # to keep cached payload minimal and compatible with older tests/consumers.
        # Fallback to calling model_dump() without kwargs for simple stand-ins used in tests.
        try:
            metriq_job.result_data = result.model_dump(exclude={"score"})
        except TypeError:
            # Some mocks (e.g., SimpleNamespace with model_dump as lambda) may not accept kwargs
            metriq_job.result_data = result.model_dump()
        job_manager.update_job(metriq_job)
        return FetchResultOutput(result=result, raw_counts=raw_counts, from_cache=False)
    else:
        print("Job is not yet completed. Current status of tasks:")
        for task in quantum_jobs:
            info = job_status(task)
            msg = f"- {task.id}: {info.status.value}"
            if info.queue_position is not None:
                msg += f" (position {info.queue_position})"
            print(msg)
        print("Please try again later.")
        return None


def view_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args.job_id, job_manager)
    if metriq_job:
        print(metriq_job)


def view_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    if not args.suite_id:
        print("Suite ID is required to view suite jobs.")
        return
    jobs = job_manager.get_jobs_by_suite_id(args.suite_id)
    if not jobs:
        print(f"No jobs found for suite ID {args.suite_id}.")
        return
    print(f"Jobs for suite ID {args.suite_id}:")
    list_jobs(jobs, show_index=False, show_suite_id=False)


def delete_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args.job_id, job_manager)
    if metriq_job:
        try:
            job_manager.delete_job(metriq_job.id)
            print(f"Job {metriq_job.id} deleted successfully.")
        except ValueError:
            print(f"Error: Job {metriq_job.id} could not be deleted. It may not exist.")
        except Exception as e:
            print(f"An unexpected error occurred while deleting the job: {e}")
    else:
        print("No job selected for deletion.")


def delete_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    if not args.suite_id:
        print("Suite ID is required to delete suite jobs.")
        return
    jobs = job_manager.get_jobs_by_suite_id(args.suite_id)
    if not jobs:
        print(f"No jobs found for suite ID {args.suite_id}.")
        return
    for job in jobs:
        try:
            job_manager.delete_job(job.id)
            print(f"Job {job.id} deleted successfully.")
        except Exception as e:
            print(f"Failed to delete job {job.id}: {e}")
    print(f"All jobs for suite ID {args.suite_id} deleted successfully.")


def estimate_job(args: argparse.Namespace, _job_manager: JobManager | None = None) -> None:
    if not args.provider:
        print("Provider is required for resource estimation.")
        return

    device = None
    if args.device:
        try:
            device = setup_device(args.provider, args.device)
        except QBraidSetupError:
            return
    else:
        print("No device specified; estimating resources without device-specific topology.")

    config_file = args.config

    if not os.path.exists(config_file):
        print(f"✗ {config_file}: Configuration file not found")
        return

    params = load_and_validate(config_file)

    available_benchmarks = _lazy_registry().get_available_benchmarks()
    if params.benchmark_name not in available_benchmarks:
        print(
            f"✗ {config_file}: Unsupported benchmark '{params.benchmark_name}'. Available: {available_benchmarks}"
        )
        return

    job_type = JobType(params.benchmark_name)
    benchmark: Benchmark = setup_benchmark(args, params, job_type)

    try:
        circuit_batches: list[CircuitBatch] = benchmark.estimate_resources_handler(device)
        resource_estimate = aggregate_resource_estimates(
            circuit_batches, hqc_fn=quantinuum_hqc_formula
        )
    except (ValueError, NotImplementedError) as exc:
        print(f"✗ {job_type.value}: {exc}")
        return
    except Exception as exc:  # pragma: no cover - surface unexpected errors cleanly
        print(f"✗ Failed to estimate resources: {exc}")
        return

    print_resource_estimate(job_type, args.provider, args.device, resource_estimate)


def main() -> int:
    load_dotenv()
    typer_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
