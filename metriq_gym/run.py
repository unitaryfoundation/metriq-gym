import argparse
from dataclasses import asdict
from datetime import datetime
import os
import sys
import logging
import uuid
from dotenv import load_dotenv

from qbraid import QbraidError
from qbraid.runtime import (
    get_providers,
    GateModelResultData,
    JobStatus,
    QuantumDevice,
    QuantumProvider,
    load_job,
    load_provider,
)
from tabulate import tabulate

from metriq_gym.exporters.dict_exporter import DictExporter
from metriq_gym.registry import (
    BENCHMARK_DATA_CLASSES,
    BENCHMARK_HANDLERS,
    BENCHMARK_RESULT_CLASSES,
    get_available_benchmarks,
)
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.cli import list_jobs, parse_arguments, prompt_for_job
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.exporters.cli_exporter import CliExporter
from metriq_gym.exporters.json_exporter import JsonExporter
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.qplatform.job import job_status
from metriq_gym.schema_validator import load_and_validate, validate_and_create_model
from metriq_gym.constants import JobType
from metriq_gym.suite_parser import parse_suite_file

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("metriq_gym")

available_benchmarks = get_available_benchmarks()

COMMON_SUITE_KEYS = ["provider", "device", "timestamp", "app_version"]


def setup_device(provider_name: str, backend_name: str) -> QuantumDevice:
    """
    Setup a QBraid device with id backend_name from specified provider.

    Args:
        provider_name: a metriq-gym supported provider name.
        backend_name: the id of a device supported by the provider.
    Raises:
        QBraidSetupError: If no device matching the name is found in the provider.
    """
    try:
        provider: QuantumProvider = load_provider(provider_name)
    except QbraidError:
        logger.error(f"No provider matching the name '{provider_name}' found.")
        logger.error(f"Providers available: {get_providers()}")
        raise QBraidSetupError("Provider not found")

    try:
        device = provider.get_device(backend_name)
    except QbraidError:
        logger.error(
            f"No device matching the name '{backend_name}' found in provider '{provider_name}'."
        )
        logger.error(f"Devices available: {[device.id for device in provider.get_devices()]}")
        raise QBraidSetupError("Device not found")
    return device


def setup_benchmark(args, params, job_type: JobType) -> Benchmark:
    return BENCHMARK_HANDLERS[job_type](args, params)


def setup_job_data_class(job_type: JobType) -> type[BenchmarkData]:
    return BENCHMARK_DATA_CLASSES[job_type]


def setup_benchmark_result_class(job_type: JobType) -> type[BenchmarkResult]:
    return BENCHMARK_RESULT_CLASSES[job_type]


def dispatch_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch a single benchmark configuration to a quantum device.

    Args:
        args: CLI arguments with benchmark config, provider, and device
        job_manager: Tracks dispatched jobs for later polling

    Note: Continues processing remaining configs if individual configs fail.
    """
    print(f"Starting dispatch on {args.provider}:{args.device}...")

    try:
        device: QuantumDevice = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    config_file = args.config

    if not os.path.exists(config_file):
        print(f"✗ {config_file}: Configuration file not found")
        return

    # Load and validate the benchmark configuration
    params = load_and_validate(config_file)

    # Validate that the benchmark exists
    if params.benchmark_name not in available_benchmarks:
        print(
            f"✗ {config_file}: Unsupported benchmark '{params.benchmark_name}'. Available: {available_benchmarks}"
        )
        return

    job_type = JobType(params.benchmark_name)

    print(f"Dispatching {params.benchmark_name}...")

    handler: Benchmark = setup_benchmark(args, params, job_type)
    job_data: BenchmarkData = handler.dispatch_handler(device)

    job_id = job_manager.add_job(
        MetriqGymJob(
            id=str(uuid.uuid4()),
            job_type=job_type,
            params=params.model_dump(exclude_none=True),
            data=asdict(job_data),
            provider_name=args.provider,
            device_name=args.device,
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
        device: QuantumDevice = setup_device(args.provider, args.device)
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

    results = []
    successful_jobs = []

    suite_id = str(uuid.uuid4())
    for benchmark_entry in suite.benchmarks:
        try:
            params = validate_and_create_model(benchmark_entry.config)

            # Validate that the benchmark exists
            if params.benchmark_name not in available_benchmarks:
                results.append(
                    f"✗ {config_file}: Unsupported benchmark '{params.benchmark_name}'. Available: {available_benchmarks}"
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
                    id=str(uuid.uuid4()),
                    job_type=job_type,
                    params=params.model_dump(exclude_none=True),
                    data=asdict(job_data),
                    provider_name=args.provider,
                    device_name=args.device,
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
    metriq_job = prompt_for_job(args, job_manager)
    if not metriq_job:
        return
    print("Polling job...")
    result = fetch_result(metriq_job, args, job_manager)
    if result is None:
        print(f"Job {metriq_job.id} is not yet completed or has no results.")
        return
    export_job_result(args, metriq_job, result)


def poll_suite(args: argparse.Namespace, job_manager: JobManager) -> None:
    if not args.suite_id:
        print("Suite ID is required to poll suite results.")
        return
    jobs = job_manager.get_jobs_by_suite_id(args.suite_id)
    if not jobs:
        print(f"No jobs found for suite ID {args.suite_id}.")
        return
    results: list[BenchmarkResult] = []
    for metriq_job in jobs:
        result = fetch_result(metriq_job, args, job_manager)
        if result is None:
            print(f"Job {metriq_job.id} is not yet completed or has no results.")
            return
        results.append(result)
    export_suite_results(args, jobs, results)


def print_selected(d, selected_keys):
    for k in selected_keys:
        if k in d:
            print(f"{k}: {d[k]}")


def export_suite_results(args, jobs: list[MetriqGymJob], results: list[BenchmarkResult]) -> None:
    if not jobs:
        return

    records = []
    for job, result in zip(jobs, results):
        records.append(DictExporter(job, result).export() | {"params": job.params})

    if hasattr(args, "json"):
        raise NotImplementedError("JSON export of suite results is not implemented yet.")
    else:
        print("\n--- Suite Metadata ---")
        print_selected(records[0], COMMON_SUITE_KEYS)
        print("\n--- Suite Results ---")
        print(tabulate_job_results(records))


def tabulate_job_results(records, sep=" – "):
    rows = []
    metric_keys = set()
    for record in records:
        metric_keys.update(record["results"].keys())
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
        for metric in metric_keys:
            row.append(record["results"].get(metric, ""))
        rows.append(row)

    return tabulate(rows, headers=headers, floatfmt=".4g")


def export_job_result(
    args: argparse.Namespace, metriq_job: MetriqGymJob, result: BenchmarkResult
) -> None:
    if hasattr(args, "json"):
        JsonExporter(metriq_job, result).export(args.json)
    else:
        CliExporter(metriq_job, result).export()


def fetch_result(
    metriq_job: MetriqGymJob, args: argparse.Namespace, job_manager: JobManager
) -> BenchmarkResult | None:
    job_type: JobType = JobType(metriq_job.job_type)
    job_result_type = setup_benchmark_result_class(job_type)
    # Ignore cached results for qnexus to ensure we use the latest normalization
    if metriq_job.result_data is not None and metriq_job.provider_name not in {"qnexus", "quantinuum_nexus"}:
        return job_result_type.model_validate(metriq_job.result_data)

    job_data: BenchmarkData = setup_job_data_class(job_type)(**metriq_job.data)
    handler = setup_benchmark(args, validate_and_create_model(metriq_job.params), job_type)
    # Build provider-agnostic QuantumJob instances via qBraid loader
    quantum_jobs = [
        load_job(job_id, provider=metriq_job.provider_name, **asdict(job_data))
        for job_id in job_data.provider_job_ids
    ]

    # Fast path: if status says completed for all, collect results
    statuses = [task.status() for task in quantum_jobs]
    if all(st == JobStatus.COMPLETED for st in statuses):
        result_data: list[GateModelResultData] = [task.result().data for task in quantum_jobs]
        result: BenchmarkResult = handler.poll_handler(job_data, result_data, quantum_jobs)
        metriq_job.result_data = result.model_dump()
        job_manager.update_job(metriq_job)
        return result

    # Otherwise, show current status and exit.
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
    metriq_job = prompt_for_job(args, job_manager)
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
    metriq_job = prompt_for_job(args, job_manager)
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


def main() -> int:
    load_dotenv()
    args = parse_arguments()
    job_manager = JobManager()

    RESOURCE_ACTION_TABLE = {
        "suite": {
            "dispatch": dispatch_suite,
            "poll": poll_suite,
            "delete": delete_suite,
        },
        "job": {
            "dispatch": dispatch_job,
            "poll": poll_job,
            "view": view_job,
            "delete": delete_job,
        },
    }

    resource_table = RESOURCE_ACTION_TABLE.get(args.resource)
    if resource_table:
        action_handler = resource_table.get(args.action)
        if action_handler:
            action_handler(args, job_manager)
            return 0
    logging.error("Invalid command. Run with --help for usage information.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
