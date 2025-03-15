import argparse
from dataclasses import asdict
from datetime import datetime
import sys
import logging
import uuid

from dotenv import load_dotenv
from qbraid import GateModelResultData, JobStatus
from qbraid.runtime import QuantumDevice, QuantumProvider, load_job, load_provider

from metriq_gym.benchmarks import BENCHMARK_DATA_CLASSES, BENCHMARK_HANDLERS
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData
from metriq_gym.cli import list_jobs, parse_arguments
from metriq_gym.exceptions import QBraidSetupError
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.schema_validator import load_and_validate, validate_and_create_model
from metriq_gym.job_type import JobType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_device(provider_name: str, backend_name: str) -> QuantumDevice:
    """
    Setup a QBraid device with id backend_name from specified provider.

    Args:
        provider_name: a metriq-gym supported provider name.
        backend_name: the id of a device supported by the provider.
    Raises:
        QBraidSetupError: If no device matching the name is found in the provider.
    """
    # TODO: https://github.com/unitaryfund/metriq-gym/issues/259
    # Once https://github.com/qBraid/qBraid/pull/890 gets released, put a defensive approach here
    # Whenever no provider is found, print qbraid.runtime.get_providers(), and raise a QBraidSetupError
    provider: QuantumProvider = load_provider(provider_name)

    try:
        device = provider.get_device(backend_name)
    except Exception:
        logger.error(
            f"No device matching the name '{backend_name}' found in provider '{provider_name}'."
        )
        logger.info(f"Devices available: {[device.id for device in provider.get_devices()]}")
        raise QBraidSetupError("Device not found")
    return device


def setup_benchmark(args, params, job_type: JobType) -> Benchmark:
    return BENCHMARK_HANDLERS[job_type](args, params)


def setup_job_data_class(job_type: JobType) -> type[BenchmarkData]:
    return BENCHMARK_DATA_CLASSES[job_type]


def dispatch_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    logger.info("Starting job dispatch...")
    try:
        device = setup_device(args.provider, args.device)
    except QBraidSetupError:
        return

    params = load_and_validate(args.input_file)
    logger.info(f"Dispatching {params.benchmark_name} benchmark job on {args.device} device...")

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
    logger.info(f"Job dispatched with ID: {job_id}")


def poll_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    if not args.job_id:
        jobs = job_manager.get_jobs()
        if not jobs:
            print("No jobs available for polling.")
            return
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
        args.job_id = jobs[selected_index].id

    logger.info("Polling job...")
    metriq_job: MetriqGymJob = job_manager.get_job(args.job_id)
    job_type: JobType = JobType(metriq_job.job_type)
    job_data: BenchmarkData = setup_job_data_class(job_type)(**metriq_job.data)
    handler = setup_benchmark(args, validate_and_create_model(metriq_job.params), job_type)
    quantum_jobs = [
        load_job(job_id, provider=metriq_job.provider_name, **asdict(job_data))
        for job_id in job_data.provider_job_ids
    ]
    if all(task.status() == JobStatus.COMPLETED for task in quantum_jobs):
        result_data: list[GateModelResultData] = [task.result().data for task in quantum_jobs]
        print(handler.poll_handler(job_data, result_data, quantum_jobs))
    else:
        logger.info("Job is not yet completed. Please try again later.")


def main() -> int:
    """Main entry point for the CLI."""
    load_dotenv()
    args = parse_arguments()
    job_manager = JobManager()

    if args.action == "dispatch":
        dispatch_job(args, job_manager)
    elif args.action == "poll":
        poll_job(args, job_manager)
    elif args.action == "list-jobs":
        jobs: list[MetriqGymJob] = job_manager.get_jobs()
        list_jobs(jobs)
    else:
        logging.error("Invalid action specified. Run with --help for usage information.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
