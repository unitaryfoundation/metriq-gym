from abc import ABC, abstractmethod
from typing import Any

from metriq_gym.benchmarks.benchmark import BenchmarkResult, BenchmarkScore
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
        # Preserve existing top-level fields.
        # For uploads/exports, include the full result payload (already contains score)
        # and also surface scalar uncertainties for convenience.
        results_block = self.result.model_dump()
        if results_block.get("score") is None:
            results_block.pop("score", None)
        result_uncertainties = dict(self.result.uncertainties or {})
        score_val = getattr(self.result, "score", None)
        if isinstance(score_val, BenchmarkScore):
            results_block["score"] = score_val.model_dump()
            result_uncertainties["score"] = score_val.uncertainty
        elif score_val is not None:
            raise TypeError("score must be a BenchmarkScore or None")
        if result_uncertainties:
            results_block["uncertainties"] = result_uncertainties
        record = {
            "app_version": self.metriq_gym_job.app_version,
            "timestamp": self.metriq_gym_job.dispatch_time.isoformat(),
            "suite_id": self.metriq_gym_job.suite_id,
            "job_type": self.metriq_gym_job.job_type.value,
            "results": results_block,
        }

        runtime_seconds = getattr(self.metriq_gym_job, "runtime_seconds", None)
        if runtime_seconds is not None:
            record["runtime_seconds"] = runtime_seconds

        # Richer suite metadata for downstream aggregation (e.g., metriq-data)
        suite_meta: dict[str, Any] = {}
        if self.metriq_gym_job.suite_id:
            suite_meta["id"] = self.metriq_gym_job.suite_id
        if self.metriq_gym_job.suite_name:
            suite_meta["name"] = self.metriq_gym_job.suite_name
        if suite_meta:
            record["suite"] = suite_meta

        job_platform = getattr(self.metriq_gym_job, "platform", None)
        platform_info: dict[str, Any] = {}
        if isinstance(job_platform, dict):
            platform_info = {k: v for k, v in job_platform.items() if v is not None}

        platform_info.setdefault("provider", self.metriq_gym_job.provider_name)
        platform_info.setdefault("device", self.metriq_gym_job.device_name)

        # Normalize AWS/Braket device identifiers for upload: use last two ARN path
        # segments, joined by underscore and lowercased (e.g., iqm_emerald).
        provider_val = str(platform_info.get("provider") or "").strip()
        device_val = str(platform_info.get("device") or "").strip()

        def _is_aws_provider(name: str) -> bool:
            return name.lower() == "aws"

        def _simplify_arn_device(device: str) -> str:
            # Split on '/' and take the last two non-empty segments when available.
            parts = [p for p in device.split("/") if p]
            if len(parts) >= 2:
                simplified = f"{parts[-2]}_{parts[-1]}"
            else:
                simplified = device
            return simplified.lower()

        if provider_val and _is_aws_provider(provider_val) and device_val:
            platform_info["device"] = _simplify_arn_device(device_val)

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
