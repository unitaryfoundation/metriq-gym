from abc import ABC, abstractmethod
from typing import Any

from metriq_gym.benchmarks.benchmark import BenchmarkResult
from metriq_gym.job_manager import MetriqGymJob


class BaseExporter(ABC):
    def __init__(self, metriq_gym_job: MetriqGymJob, result: BenchmarkResult):
        self.metriq_gym_job = metriq_gym_job
        self.result = result
        super().__init__()

    def _derive_device_metadata(self) -> dict[str, Any]:
        """Use metadata collected at dispatch time on the job object."""
        try:
            plat = getattr(self.metriq_gym_job, "platform", None) or {}
            md = plat.get("device_metadata", {}) if isinstance(plat, dict) else {}
            return md if isinstance(md, dict) else {}
        except Exception:
            return {}

    def as_dict(self):
        # Preserve existing top-level fields for backward compatibility.
        record = {
            "app_version": self.metriq_gym_job.app_version,
            "timestamp": self.metriq_gym_job.dispatch_time.isoformat(),
            "provider": self.metriq_gym_job.provider_name,
            "suite_id": self.metriq_gym_job.suite_id,
            "device": self.metriq_gym_job.device_name,
            "job_type": self.metriq_gym_job.job_type.value,
            "results": dict(self.result),
        }

        platform = {
            "provider": self.metriq_gym_job.provider_name,
            "device": self.metriq_gym_job.device_name,
            "device_metadata": self._derive_device_metadata(),
        }
        record["platform"] = platform

        return record

    @abstractmethod
    def export(self, *args, **kwargs):
        pass
