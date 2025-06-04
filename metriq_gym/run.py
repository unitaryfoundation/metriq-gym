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

from metriq_gym.benchmarks import BENCHMARK_DATA_CLASSES, BENCHMARK_HANDLERS
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
    """Get the path to the example file for a given benchmark type.

    Maps each benchmark type to its corresponding example configuration file
    in the schemas/examples/ directory. These example files contain predefined
    parameters for running benchmarks in multi-benchmark dispatch mode.

    Args:
        job_type: The benchmark type to get the example file for.

    Returns:
        Full path to the example configuration file for the specified benchmark.

    Supported benchmarks:
        - BSEQ: Bell State Effective Qubits benchmark
        - CLOPS: Circuit Layer Operations per Second benchmark
        - QML_KERNEL: Quantum Machine Learning Kernel benchmark
        - QUANTUM_VOLUME: Quantum Volume benchmark
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    example_files = {
        JobType.BSEQ: "bseq.example.json",
        JobType.CLOPS: "clops.example.json",
        JobType.QML_KERNEL: "qml_kernel.example.json",
        JobType.QUANTUM_VOLUME: "quantum_volume.example.json",
    }
    return os.path.join(current_dir, "schemas", "examples", example_files[job_type])


def dispatch_all_benchmarks(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch all available benchmarks to a single device.

    This function orchestrates the execution of multiple benchmarks on a single
    quantum device, providing comprehensive device characterization capabilities.
    Each benchmark runs with predefined parameters from example configuration files.

    Features:
        - Runs all available benchmarks using predefined example configurations
        - Supports benchmark exclusion via --except flag
        - Provides detailed progress reporting and error handling
        - Continues execution even if individual benchmarks fail
        - Generates comprehensive success/failure summary

    Args:
        args: Command-line arguments containing:
            - provider: Quantum provider name (e.g., 'ibm', 'aws')
            - device: Device identifier (e.g., 'ibm_sherbrooke')
            - exclude_benchmarks: Optional list of benchmarks to exclude
        job_manager: JobManager instance for tracking dispatched jobs

    Workflow:
        1. Setup quantum device connection
        2. Determine which benchmarks to run (all minus excluded)
        3. For each benchmark:
           - Load example configuration file
           - Setup benchmark handler
           - Dispatch job to device
           - Record success/failure
        4. Display comprehensive summary with job IDs

    Error handling:
        - Device setup failures: Return early with error message
        - Missing example files: Skip benchmark with warning
        - Benchmark failures: Log error and continue with remaining benchmarks
    """
    print("Starting bulk benchmark dispatch...")

    try:
        device = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    # Get all available benchmarks
    available_benchmarks = list(JobType)

    # Filter excluded benchmarks
    if args.exclude_benchmarks:
        excluded = [
            JobType(name)
            for name in args.exclude_benchmarks
            if name in [jt.value for jt in JobType]
        ]
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

    print("\nSummary:")
    for result in results:
        print(f"  {result}")

    print(
        f"\nSuccessfully dispatched {len(successful_jobs)}/{len(available_benchmarks)} benchmarks."
    )
    if successful_jobs:
        print("Use 'mgym poll' to check job status.")


def dispatch_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    """Dispatch job(s) based on provided arguments.

    This function supports both single benchmark and multi-benchmark dispatch modes,
    acting as the main entry point for job dispatching. The dispatch mode is determined
    by the presence of the --all-benchmarks flag in the command-line arguments.

    Dispatch modes:
        1. Multi-benchmark mode (--all-benchmarks flag present):
           - Routes to dispatch_all_benchmarks()
           - Runs all available benchmarks using example configurations
           - Supports benchmark exclusion via --except

        2. Single benchmark mode (traditional behavior):
           - Requires input_file with benchmark configuration
           - Dispatches single benchmark as specified in the file
           - Maintains full backward compatibility

    Args:
        args: Command-line arguments containing:
            - all_benchmarks: Boolean flag for multi-benchmark mode
            - input_file: Path to benchmark configuration (required for single mode)
            - provider: Quantum provider name
            - device: Device identifier
        job_manager: JobManager instance for tracking dispatched jobs

    Validation:
        - Ensures input_file is provided when not using --all-benchmarks
        - Validates device and provider accessibility
        - Handles configuration file loading and validation

    Error handling:
        - Missing input_file: Logs error and returns
        - Device setup failures: Handled by setup_device()
        - Invalid configurations: Handled by load_and_validate()
    """
    if args.all_benchmarks:
        dispatch_all_benchmarks(args, job_manager)
        return

    # Single benchmark dispatch logic
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
