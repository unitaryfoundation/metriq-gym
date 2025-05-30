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

from metriq_gym.benchmarks import BENCHMARK_DATA_CLASSES, BENCHMARK_HANDLERS, SCHEMA_MAPPING
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData
from metriq_gym.cli import parse_arguments, prompt_for_job
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.exporters.cli_exporter import CliExporter
from metriq_gym.exporters.json_exporter import JsonExporter
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.schema_validator import load_and_validate, validate_and_create_model
from metriq_gym.benchmarks import JobType

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


def get_example_file_path(job_type: JobType) -> str:
    """Get the path to the example file for a given benchmark type."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    example_files = {
        JobType.BSEQ: "bseq.example.json",
        JobType.CLOPS: "clops.example.json", 
        JobType.QML_KERNEL: "qml_kernel.example.json",
        JobType.QUANTUM_VOLUME: "quantum_volume.example.json",
    }
    return os.path.join(current_dir, "schemas", "examples", example_files[job_type])


def dispatch_all_benchmarks(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch all available benchmarks to a single device."""
    print("Starting bulk benchmark dispatch...")
    
    try:
        device = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    # Get all available benchmarks
    available_benchmarks = list(JobType)
    
    # Filter excluded benchmarks
    if args.exclude_benchmarks:
        excluded = [JobType(name) for name in args.exclude_benchmarks if name in [jt.value for jt in JobType]]
        available_benchmarks = [jt for jt in available_benchmarks if jt not in excluded]
        if args.exclude_benchmarks:
            print(f"Excluding benchmarks: {[jt.value for jt in excluded]}")

    print(f"Running {len(available_benchmarks)} benchmarks on {args.device} device...")
    
    results = []
    successful_jobs = []
    
    for job_type in available_benchmarks:
        try:
            # Load example file for this benchmark
            example_file = get_example_file_path(job_type)
            if not os.path.exists(example_file):
                results.append(f"✗ {job_type.value}: Example file not found")
                continue
                
            params = load_and_validate(example_file)
            handler = setup_benchmark(args, params, job_type)
            job_data = handler.dispatch_handler(device)
            
            job_id = job_manager.add_job(
                MetriqGymJob(
                    id=str(uuid.uuid4()),
                    job_type=job_type,
                    params=params.model_dump(),
                    data=asdict(job_data),
                    provider_name=args.provider,
                    device_name=args.device,
                    dispatch_time=datetime.now(),
                )
            )
            
            results.append(f"✓ {job_type.value} dispatched with ID: {job_id}")
            successful_jobs.append(job_id)
            
        except Exception as e:
            results.append(f"✗ {job_type.value} failed: {str(e)}")
    
    # Print summary
    print("\nSummary:")
    for result in results:
        print(f"  {result}")
    
    print(f"\nSuccessfully dispatched {len(successful_jobs)}/{len(available_benchmarks)} benchmarks.")
    if successful_jobs:
        print("Use 'mgym poll' to check job status.")


def dispatch_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch job(s) based on provided arguments."""
    if args.all_benchmarks:
        dispatch_all_benchmarks(args, job_manager)
        return
    
    # Original single benchmark dispatch logic
    if not args.input_file:
        logger.error("input_file is required when not using --all-benchmarks")
        return
        
    print("Starting job dispatch...")
    try:
        device = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    params = load_and_validate(args.input_file)
    print(f"Dispatching {params.benchmark_name} benchmark job on {args.device} device...")

    job_type = JobType(params.benchmark_name)
    handler: Benchmark = setup_benchmark(args, params, job_type)
    job_data: BenchmarkData = handler.dispatch_handler(device)
    job_id = job_manager.add_job(
        MetriqGymJob(
            id=str(uuid.uuid4()),
            job_type=job_type,
            params=params.model_dump(),
            data=asdict(job_data),
            provider_name=args.provider,
            device_name=args.device,
            dispatch_time=datetime.now(),
        )
    )
    print(f"Job dispatched with ID: {job_id}")


def poll_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args, job_manager)
    if not metriq_job:
        return
    print("Polling job...")
    job_type: JobType = JobType(metriq_job.job_type)
    job_data: BenchmarkData = setup_job_data_class(job_type)(**metriq_job.data)
    handler = setup_benchmark(args, validate_and_create_model(metriq_job.params), job_type)
    quantum_jobs = [
        load_job(job_id, provider=metriq_job.provider_name, **asdict(job_data))
        for job_id in job_data.provider_job_ids
    ]
    if all(task.status() == JobStatus.COMPLETED for task in quantum_jobs):
        result_data: list[GateModelResultData] = [task.result().data for task in quantum_jobs]
        results = handler.poll_handler(job_data, result_data, quantum_jobs)
        if hasattr(args, "json"):
            JsonExporter(metriq_job, results).export(args.json)
        else:
            CliExporter(metriq_job, results).export()
    else:
        print("Job is not yet completed. Please try again later.")


def view_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    metriq_job = prompt_for_job(args, job_manager)
    if metriq_job:
        print(metriq_job)


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
    else:
        logging.error("Invalid action specified. Run with --help for usage information.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())