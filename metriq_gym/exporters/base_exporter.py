from abc import ABC, abstractmethod
from metriq_gym.benchmarks.benchmark import BenchmarkResult
from metriq_gym.job_manager import MetriqGymJob


class BaseExporter(ABC):
    def __init__(self, metriq_gym_job: MetriqGymJob, result: BenchmarkResult):
        self.metriq_gym_job = metriq_gym_job
        self.result = result
        super().__init__()

    def as_dict(self):
        return {
            "app_version": self.metriq_gym_job.app_version,
            "timestamp": self.metriq_gym_job.dispatch_time.isoformat(),
            "provider": self.metriq_gym_job.provider_name,
            "suite_id": self.metriq_gym_job.suite_id,
            "device": self.metriq_gym_job.device_name,
            "job_type": self.metriq_gym_job.job_type.value,
            "results": dict(self.result),
        }

    @abstractmethod
    def export(self, *args, **kwargs):
        pass
