from abc import ABC, abstractmethod
import importlib.metadata
from metriq_gym.benchmarks.benchmark import BenchmarkResult
from metriq_gym.job_manager import MetriqGymJob


class BaseExporter(ABC):
    def __init__(self, metriq_gym_job: MetriqGymJob, results: BenchmarkResult):
        self.metriq_gym_job = metriq_gym_job
        self.results = results
        super().__init__()

    def _as_dict(self):
        return {
            "version": importlib.metadata.version("metriq-gym"),
            "timestamp": self.metriq_gym_job.dispatch_time.isoformat(),
            "provider": self.metriq_gym_job.provider_name,
            "device": self.metriq_gym_job.device_name,
            "job_type": self.metriq_gym_job.job_type.value,
            "results": dict(self.results),
        }

    @abstractmethod
    def export(self, *args, **kwargs):
        pass
