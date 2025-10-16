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
        results_payload = self.result.result_metrics()
        uncertainties_payload = self.result.uncertainty_metrics()

        record = {
            "app_version": self.metriq_gym_job.app_version,
            "timestamp": self.metriq_gym_job.dispatch_time.isoformat(),
            "suite_id": self.metriq_gym_job.suite_id,
            "job_type": self.metriq_gym_job.job_type.value,
            "results": {
                "values": results_payload,
                "uncertainties": uncertainties_payload if uncertainties_payload else {},
            },
        }

        job_platform = getattr(self.metriq_gym_job, "platform", None)
        platform_info: dict[str, Any]
        if isinstance(job_platform, dict):
            platform_info = {k: v for k, v in job_platform.items() if v is not None}
        else:
            platform_info = {}

        platform_info.setdefault("provider", self.metriq_gym_job.provider_name)
        platform_info.setdefault("device", self.metriq_gym_job.device_name)

        device_metadata = self._derive_device_metadata()
        if device_metadata:
            platform_info["device_metadata"] = device_metadata
        elif "device_metadata" in platform_info and not platform_info["device_metadata"]:
            platform_info.pop("device_metadata")

        record["platform"] = platform_info

        return record

    @abstractmethod
    def export(self, *args, **kwargs):
        pass
