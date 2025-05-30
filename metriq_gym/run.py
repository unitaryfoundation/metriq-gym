
"""Command-line parsing for running metriq benchmarks."""

import argparse
from dataclasses import asdict
from datetime import datetime
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


# Device name mapping for backward compatibility
DEVICE_MAPPING = {
    "aer_simulator": "qiskit.aer.automatic",
    "aer_simulator_statevector": "qiskit.aer.statevector", 
    "aer_simulator_stabilizer": "qiskit.aer.stabilizer",
    "aer_simulator_density_matrix": "qiskit.aer.density_matrix",
    "qrack": "qrack.cpu",
    "qrack_gpu": "qrack.gpu",
}


class LocalProvider:
    """Local provider for managing local quantum simulators.
    
    This provider uses a plugin architecture to support multiple simulator backends
    through a unified interface, making it extensible and maintainable.
    """
    
    def __init__(self):
        self._load_available_devices()
    
    def _load_available_devices(self):
        """Load available devices from configuration and adapters."""
        from metriq_gym.qplatform.device import get_available_simulators
        self._available_devices = get_available_simulators()
    
    def get_device(self, device_id: str):
        """Get a local device by ID with backward compatibility mapping.
        
        Args:
            device_id: Device identifier (e.g., 'aer_simulator' or 'qiskit.aer.statevector')
            
        Returns:
            LocalDevice: A local device wrapper
            
        Raises:
            QBraidSetupError: If device is not supported
        """
        # Apply backward compatibility mapping
        mapped_device_id = DEVICE_MAPPING.get(device_id, device_id)
        
        if mapped_device_id not in self._available_devices:
            available = list(self._available_devices.keys())
            legacy_available = list(DEVICE_MAPPING.keys())
            all_available = sorted(set(available + legacy_available))
            logger.error(f"Local device '{device_id}' not supported.")
            logger.error(f"Available local devices: {all_available}")
            raise QBraidSetupError(f"Local device '{device_id}' not found")
        
        from metriq_gym.qplatform.device import create_local_device
        return create_local_device(mapped_device_id)
    
    def get_devices(self):
        """Get all available local devices.
        
        Returns:
            list: List of available LocalDevice instances
        """
        from metriq_gym.qplatform.device import create_local_device
        devices = []
        for device_id in self._available_devices.keys():
            try:
                device = create_local_device(device_id)
                devices.append(device)
            except Exception as e:
                logger.warning(f"Failed to create device {device_id}: {e}")
                continue
        return devices


def setup_device(provider_name: str, backend_name: str) -> QuantumDevice:
    """
    Setup a QBraid device with id backend_name from specified provider.
    
    Supports both remote providers (via qBraid) and local simulators.

    Args:
        provider_name: a metriq-gym supported provider name ('local' for local simulators).
        backend_name: the id of a device supported by the provider.
        
    Returns:
        QuantumDevice: A device instance compatible with qBraid interface
        
    Raises:
        QBraidSetupError: If no device matching the name is found in the provider.
    """
    # Handle local simulators
    if provider_name == "local":
        try:
            provider = LocalProvider()
            device = provider.get_device(backend_name)
            logger.info(f"Successfully configured local device: {backend_name}")
            return device
        except QBraidSetupError:
            raise
        except Exception as e:
            logger.error(f"Failed to setup local device '{backend_name}': {e}")
            raise QBraidSetupError(f"Local device setup failed: {e}")
    
    # Handle remote providers (existing code unchanged)
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
    
    # Handle local jobs differently from remote jobs
    if metriq_job.provider_name == "local":
        from metriq_gym.qplatform.job import load_local_job
        quantum_jobs = [
            load_local_job(job_id, **asdict(job_data))
            for job_id in job_data.provider_job_ids
        ]
    else:
        # Remote jobs (existing behavior)
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
