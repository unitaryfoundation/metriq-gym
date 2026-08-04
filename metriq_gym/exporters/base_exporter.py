from abc import ABC, abstractmethod
from typing import Any

from metriq_gym.benchmarks.benchmark import BenchmarkResult
from metriq_gym.job_manager import MetriqGymJob
from metriq_gym.platform import canonical_device_name, canonical_provider_name


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
        results_block = self.result.model_dump()
        if results_block.get("score") is None:
            results_block.pop("score", None)
        # Do not emit a separate uncertainties block; structured fields carry their own
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

        provider_val = canonical_provider_name(str(platform_info.get("provider") or ""))
        device_val = canonical_device_name(
            provider_val,
            str(platform_info.get("device") or ""),
        )
        platform_info["provider"] = provider_val
        platform_info["device"] = device_val

        device_metadata = self._derive_device_metadata()
        if device_metadata:
            platform_info["device_metadata"] = device_metadata
        elif "device_metadata" in platform_info and not platform_info["device_metadata"]:
            platform_info.pop("device_metadata")

        record["platform"] = platform_info

        # Surface per-circuit two-qubit gate counts collected at dispatch time.
        # These live on the BenchmarkData (stored as the job's ``data`` dict) and
        # are emitted additively so existing fields are unaffected.
        job_data = getattr(self.metriq_gym_job, "data", None)
        if isinstance(job_data, dict):
            circuit_metadata = {
                key: job_data[key]
                for key in (
                    "input_two_qubit_gate_counts",
                    "transpiled_two_qubit_gate_counts",
                )
                if job_data.get(key) is not None
            }
            if circuit_metadata:
                record["circuit_metadata"] = circuit_metadata

        return record

    @abstractmethod
    def export(self, *args, **kwargs):
        pass
