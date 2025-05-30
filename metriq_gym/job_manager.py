from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
import pprint
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

    def _log_skip(self, line_number: int, reason: str) -> None:
        """Log a warning for skipped invalid job entry."""
        logger.warning(f"Skipping job on line {line_number} in {self.jobs_file}: {reason}")

    def _load_jobs(self):
        """Load jobs from the local JSONL database file, skipping invalid entries.
        
        This method loads all valid jobs from the JSONL file and silently skips
        any invalid entries (e.g., due to schema changes, corrupted data, or
        unsupported job types). Invalid jobs are logged as warnings but do not
        prevent the loading of other valid jobs.
        
        Invalid jobs can occur when:
        - The job type no longer exists in the current JobType enum
        - The datetime format is invalid
        - Required fields are missing
        - The JSON structure doesn't match the expected schema
        - The file contains malformed JSON
        """
        self.jobs = []
        if not os.path.exists(self.jobs_file):
            return
        
        with open(self.jobs_file) as file:
            for line_number, raw in enumerate(file, 1):
                line = raw.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    job = MetriqGymJob.deserialize(line)
                except json.JSONDecodeError as e:
                    self._log_skip(line_number, f"Malformed JSON at pos {e.pos}")
                except KeyError as e:
                    self._log_skip(line_number, f"Missing field {e}")
                except ValueError as e:
                    text = str(e).lower()
                    if "not a valid" in text and "jobtype" in text:
                        self._log_skip(line_number, f"Unknown job type: {e}")
                    elif "datetime" in text or "time" in text:
                        self._log_skip(line_number, f"Bad datetime format: {e}")
                    else:
                        self._log_skip(line_number, f"Invalid value: {e}")
                except TypeError as e:
                    self._log_skip(line_number, f"Incorrect data structure: {e}")
                except Exception as e:
                    self._log_skip(line_number, f"Unexpected error ({type(e).__name__}): {e}")
                else:
                    # Only append if no exception occurred
                    self.jobs.append(job)

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
