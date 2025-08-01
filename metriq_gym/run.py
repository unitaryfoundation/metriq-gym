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

from metriq_gym.registry import (
    BENCHMARK_DATA_CLASSES,
    BENCHMARK_HANDLERS,
    get_available_benchmarks,
)
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData
from metriq_gym.cli import parse_arguments, prompt_for_job
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.exporters.cli_exporter import CliExporter
from metriq_gym.exporters.json_exporter import JsonExporter
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.qplatform.job import job_status
from metriq_gym.schema_validator import load_and_validate, validate_and_create_model
from metriq_gym.constants import JobType

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("metriq_gym")


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


def dispatch_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch multiple benchmark configurations to a quantum device.

    Enables comprehensive device characterization by running the same benchmark
    type with different parameters or multiple benchmark types in sequence.

    Args:
        args: CLI arguments with benchmark configs, provider, and device
        job_manager: Tracks dispatched jobs for later polling

    Note: Continues processing remaining configs if individual configs fail.
    """
    print("Starting job dispatch...")

    try:
        device: QuantumDevice = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    results = []
    successful_jobs = []

    for config_file in args.benchmark_configs:
        try:
            if not os.path.exists(config_file):
                results.append(f"✗ {config_file}: Configuration file not found")
                continue

            params = load_and_validate(config_file)

            # Validate that the benchmark exists
            available_benchmarks = get_available_benchmarks()
            if params.benchmark_name not in available_benchmarks:
                results.append(
                    f"✗ {config_file}: Unsupported benchmark '{params.benchmark_name}'. Available: {available_benchmarks}"
                )
                continue

            job_type = JobType(params.benchmark_name)

            print(
                f"Dispatching {params.benchmark_name} benchmark from {config_file} on {args.device}..."
            )

            handler: Benchmark = setup_benchmark(args, params, job_type)
            job_data: BenchmarkData = handler.dispatch_handler(device)

            job_dict = asdict(job_data)
            job_dict["cache"] = None

            job_id = job_manager.add_job(
                MetriqGymJob(
                    id=str(uuid.uuid4()),
                    job_type=job_type,
                    params=params.model_dump(exclude_none=True),
                    data=job_dict,
                    provider_name=args.provider,
                    device_name=args.device,
                    dispatch_time=datetime.now(),
                )
            )

            results.append(
                f"✓ {params.benchmark_name} ({config_file}) dispatched with ID: {job_id}"
            )
            successful_jobs.append(job_id)

        except Exception as e:
            error_details = f"{type(e).__name__}: {str(e)}"
            results.append(f"✗ {config_file} failed: {error_details}")

    print("\nSummary:")
    for result in results:
        print(f"  {result}")

    print(
        f"\nSuccessfully dispatched {len(successful_jobs)}/{len(args.benchmark_configs)} benchmarks."
    )
    if successful_jobs:
        print("Use 'mgym poll' to check job status.")


def poll_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args, job_manager)
    if not metriq_job:
        return
    print("Polling job...")
    job_type: JobType = JobType(metriq_job.job_type)
    job_data_dict = dict(metriq_job.data)

    cache = job_data_dict.pop("cache", None)
    job_data: BenchmarkData = setup_job_data_class(job_type)(**job_data_dict)

    handler = setup_benchmark(args, validate_and_create_model(metriq_job.params), job_type)

    if cache:
        cache_result_data = [GateModelResultData.from_dict(c) for c in cache]
        results = handler.poll_handler(job_data, cache_result_data, [])
        if hasattr(args, "json"):
            JsonExporter(metriq_job, results).export(args.json)
        else:
            CliExporter(metriq_job, results).export()
        return

    quantum_jobs = [
        load_job(job_id, provider=metriq_job.provider_name, **asdict(job_data))
        for job_id in job_data.provider_job_ids
    ]

    if all(task.status() == JobStatus.COMPLETED for task in quantum_jobs):
        result_data: list[GateModelResultData] = [task.result().data for task in quantum_jobs]
        metriq_job.data["cache"] = [r.to_dict() for r in result_data]
        job_manager.update_job(metriq_job)
        results = handler.poll_handler(job_data, result_data, quantum_jobs)
        if hasattr(args, "json"):
            JsonExporter(metriq_job, results).export(args.json)
        else:
            CliExporter(metriq_job, results).export()
    else:
        print("Job is not yet completed. Current status:")
        for task in quantum_jobs:
            info = job_status(task)
            msg = f"- {task.id}: {info.status.value}"
            if info.queue_position is not None:
                msg += f" (position {info.queue_position})"
            print(msg)
        print("Please try again later.")


def view_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args, job_manager)
    if metriq_job:
        print(metriq_job)


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


def main() -> int:
    """Main entry point for the CLI."""
    load_dotenv()
    args = parse_arguments()
    job_manager = JobManager()

    if args.action == "dispatch":
        dispatch_job(args, job_manager)
    elif args.action == "view":
        view_job(args, job_manager)
    elif args.action == "poll":
        poll_job(args, job_manager)
    elif args.action == "delete":
        delete_job(args, job_manager)
    else:
        logging.error("Invalid action specified. Run with --help for usage information.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
