import os
from metriq_client import MetriqClient
from metriq_client.models import ResultCreateRequest

from metriq_gym.benchmarks.benchmark import BenchmarkData, BenchmarkResult
from metriq_gym.job_manager import MetriqGymJob
from metriq_gym.job_type import JobType
from metriq_gym.exporters.base_exporter import BaseExporter

METRIQ_APP_DATETIME_FORMAT = "%Y-%m-%d"

METRIQ_APP_PLATFORM_ID_MAPPING = {
    "ibm_torino": 209,
}

METRIQ_APP_TASK_ID_MAPPING = {
    JobType.BSEQ: 236,
}

# TODO: Update this with an id that represent metriq-gym as method
METRIQ_APP_METHOD_ID = 426


METRIQ_APP_SUBMISSION_ID = 800


class MetriqClientExporter(BaseExporter):
    def __init__(self):
        self.client = MetriqClient(os.environ.get("METRIQ_CLIENT_API_KEY"))

    def get_submission_id(self) -> str:
        return str(METRIQ_APP_SUBMISSION_ID)

    def get_task_id(self, job_type: JobType) -> str:
        task_id = METRIQ_APP_TASK_ID_MAPPING.get(job_type)
        if task_id is None:
            raise ValueError(f"Task ID not found for job type: {job_type}")
        return str(task_id)

    def get_method_id(self) -> str:
        return str(METRIQ_APP_METHOD_ID)

    def get_platform_id(self, provider_name: str, device_name: str) -> str:
        platform_id = METRIQ_APP_PLATFORM_ID_MAPPING.get(device_name.lower())
        if platform_id is None:
            raise ValueError(
                f"Platform ID not found for provider: {provider_name}, device: {device_name}"
            )
        return str(platform_id)

    def get_result_requests(
        self,
        results: BenchmarkResult,
        metriq_gym_job: MetriqGymJob,
        task_id: str,
        method_id: str,
    ) -> list[ResultCreateRequest]:
        """Convert benchmark result data to Metriq client requests."""
        dispatch_time_str = metriq_gym_job.dispatch_time.strftime(METRIQ_APP_DATETIME_FORMAT)
        platform_id = self.get_platform_id(metriq_gym_job.provider_name, metriq_gym_job.device_name)

        result_requests = []

        for name, field in results.model_fields.items():
            result_request = ResultCreateRequest(
                task=task_id,
                method=method_id,
                platform=platform_id,
                isHigherBetter=str(True),
                metricName=field.title or name,
                metricValue=str(getattr(results, name)),
                evaluatedAt=dispatch_time_str,
            )
            result_requests.append(result_request)
        return result_requests

    def submit(
        self, metriq_gym_job: MetriqGymJob, job_data: BenchmarkData, results: BenchmarkResult
    ) -> None:
        task_id = self.get_task_id(metriq_gym_job.job_type)
        method_id = self.get_method_id()
        submission_id = self.get_submission_id()
        platform_id = self.get_platform_id(metriq_gym_job.provider_name, metriq_gym_job.device_name)

        self.client.submission_add_task(submission_id, task_id)
        self.client.submission_add_method(submission_id, method_id)
        self.client.submission_add_platform(submission_id, platform_id)

        requests = self.get_result_requests(
            results,
            metriq_gym_job,
            task_id,
            method_id,
        )

        for req in requests:
            self.client.result_add(req, submission_id)
