import argparse
from dataclasses import asdict
from datetime import datetime
import sys
import logging
import uuid

from dotenv import load_dotenv
from qbraid import JobStatus, QuantumJob, ResultData
from qbraid.runtime import QuantumDevice, QuantumProvider

from metriq_gym.benchmarks import BENCHMARK_DATA_CLASSES, BENCHMARK_HANDLERS
from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData
from metriq_gym.cli import list_jobs, parse_arguments
from metriq_gym.job_manager import JobManager, MetriqGymJob
from metriq_gym.provider import QBRAID_PROVIDERS, ProviderType
from metriq_gym.schema_validator import load_and_validate
from metriq_gym.job_type import JobType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_device(provider_name: str, backend_name: str) -> QuantumDevice:
    provider: QuantumProvider = QBRAID_PROVIDERS[ProviderType(provider_name)]["provider"]
    device: QuantumDevice = provider().get_device(backend_name)
    return device


def setup_handler(args, params, job_type) -> type[Benchmark]:
    return BENCHMARK_HANDLERS[job_type](args, params)


def setup_job_class(provider_name: str) -> type[QuantumJob]:
    return QBRAID_PROVIDERS[ProviderType(provider_name)]["job_class"]


def setup_job_data_class(job_type: JobType) -> type[BenchmarkData]:
    return BENCHMARK_DATA_CLASSES[job_type]


def dispatch_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    logger.info("Dispatching job...")
    provider_name = args.provider
    device_name = args.device
    device = setup_device(provider_name, device_name)
    params = load_and_validate(args.input_file)
    job_type = JobType(params.benchmark_name)
    handler: Benchmark = setup_handler(args, params, job_type)
    job_data: BenchmarkData = handler.dispatch_handler(device)
    job_id = job_manager.add_job(
        MetriqGymJob(
            id=str(uuid.uuid4()),
            job_type=job_type,
            params=params.model_dump(),
            data=asdict(job_data),
            provider_name=provider_name,
            device_name=device_name,
            dispatch_time=datetime.now(),
        )
    )
    logger.info(f"Job dispatched with ID: {job_id}")


def poll_job(args: argparse.Namespace, job_manager: JobManager) -> None:
    logger.info("Polling job...")
    metriq_job: MetriqGymJob = job_manager.get_job(args.job_id)
    job_type: JobType = JobType(metriq_job.job_type)
    job_data: BenchmarkData = setup_job_data_class(job_type)(**metriq_job.data)
    job_class = setup_job_class(metriq_job.provider_name)
    device = setup_device(metriq_job.provider_name, metriq_job.device_name)
    handler = setup_handler(args, None, job_type)
    quantum_job = [job_class(job_id, device=device) for job_id in job_data.provider_job_ids]
    if all(task.status() == JobStatus.COMPLETED for task in quantum_job):
        result_data: list[ResultData] = [task.result().data for task in quantum_job]
        handler.poll_handler(job_data, result_data)
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
        list_jobs(job_manager)

    else:
        logging.error("Invalid action specified. Run with --help for usage information.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
