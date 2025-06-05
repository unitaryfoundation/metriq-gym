from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import pprint
import logging
from typing import Any

from tabulate import tabulate
from metriq_gym.benchmarks import JobType

logger = logging.getLogger(__name__)


@dataclass
class MetriqGymJob:
    id: str
    job_type: JobType
    params: dict[str, Any]
    data: dict[str, Any]
    provider_name: str
    device_name: str
    dispatch_time: datetime

    def to_table_row(self) -> list[str]:
        return [
            self.id,
            self.provider_name,
            self.device_name,
            self.job_type,
            self.dispatch_time.isoformat(),
        ]

    def serialize(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @staticmethod
    def deserialize(data: str) -> "MetriqGymJob":
        job_dict = json.loads(data)
        job = MetriqGymJob(**job_dict)
        job.job_type = JobType(job_dict["job_type"])
        job.dispatch_time = datetime.fromisoformat(job_dict["dispatch_time"])
        return job

    def __str__(self) -> str:
        rows = [
            ["id", self.id],
            ["job_type", self.job_type.value],
            ["params", pprint.pformat(self.params)],
            ["provider_name", self.provider_name],
            ["device_name", self.device_name],
            ["provider_job_ids", pprint.pformat(self.data["provider_job_ids"])],
            ["dispatch_time", self.dispatch_time.isoformat()],
        ]
        return tabulate(rows, tablefmt="fancy_grid")


# TODO: https://github.com/unitaryfoundation/metriq-gym/issues/51
class JobManager:
    jobs: list[MetriqGymJob]
    jobs_file = ".metriq_gym_jobs.jsonl"

    def __init__(self):
        self._load_jobs()

    def _load_jobs(self):
        self.jobs = []
        if os.path.exists(self.jobs_file):
            with open(self.jobs_file) as file:
                for line_number, line in enumerate(file, start=1):
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    try:
                        job = MetriqGymJob.deserialize(stripped_line)
                        
                        # Validate required fields
                        if not isinstance(getattr(job, "job_type", None), JobType):
                            raise ValueError("Invalid or missing job_type")
                            
                        if not isinstance(getattr(job, "params", None), dict):
                            raise TypeError("Invalid or missing params")
                            
                        if not isinstance(getattr(job, "device_name", None), str):
                            raise ValueError("Invalid or missing device_name")
                            
                        self.jobs.append(job)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_number}: Invalid JSON (pos {e.pos})")
                    except (KeyError, TypeError, ValueError) as e:
                        logger.warning(f"Line {line_number}: {e}")
                    except Exception as e:
                        logger.warning(f"Line {line_number}: Unexpected error ({type(e).__name__}) - {e}")

        if not self.jobs:
            logger.warning(f"No valid jobs found in {self.jobs_file}")

    def add_job(self, job: MetriqGymJob) -> str:
        self.jobs.append(job)
        with open(self.jobs_file, "a") as file:
            file.write(job.serialize() + "\n")
        return job.id

    def get_jobs(self) -> list[MetriqGymJob]:
        return self.jobs

    def get_job(self, job_id: str) -> MetriqGymJob:
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise ValueError(f"Job with id {job_id} not found")
