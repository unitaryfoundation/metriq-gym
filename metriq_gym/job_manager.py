from dataclasses import asdict, dataclass, field
from datetime import datetime
import importlib.metadata
import json
import os
import pprint
import logging
from typing import Any

from tabulate import tabulate
from metriq_gym.constants import JobType
from metriq_gym.paths import get_data_db_path


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
    version: str = field(default_factory=lambda: importlib.metadata.version("metriq-gym"))
    result_data: list[dict[str, Any]] | None = None

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
        version = job_dict.get("version", importlib.metadata.version("metriq-gym"))
        job = MetriqGymJob(**{**job_dict, "version": version})
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
    jobs_file = get_data_db_path()

    def __init__(self):
        self._load_jobs()

    def _log_skip(self, line_number: int, reason: str) -> None:
        logger.warning(f"Skipping job on line {line_number} in {self.jobs_file}: {reason}")

    def _load_jobs(self):
        """
        Initialize the job list by loading valid jobs from the local JSONL db file.

        It reads the file line by line, attempting to deserialize each entry into a `MetriqGymJob` object.
        It skips invalid or outdated entries without raising exceptions, while logging the reasons for each skip.

        Jobs may be skipped for the following reasons:
        - JSON decoding errors
        - Missing required fields
        - Invalid job types not defined in `JobType`
        - Incorrect datetime format
        - Structural mismatches or unsupported schemas

        All successfully parsed jobs are stored in `self.jobs`.
        """
        self.jobs = []

        if not os.path.exists(self.jobs_file):
            return

        with open(self.jobs_file) as file:
            for line_number, raw_line in enumerate(file, start=1):
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue  # Ignore blank lines

                try:
                    job = MetriqGymJob.deserialize(stripped_line)
                except json.JSONDecodeError as e:
                    self._log_skip(line_number, f"Invalid JSON at position {e.pos}")
                except KeyError as e:
                    self._log_skip(line_number, f"Missing required field: {e}")
                except ValueError as e:
                    message = str(e).lower()
                    if "not a valid" in message and "jobtype" in message:
                        # Attempt to extract the invalid enum value
                        invalid_value = str(e).split("'")[1] if "'" in str(e) else str(e)
                        self._log_skip(line_number, f"Unknown job type: '{invalid_value}'")
                    elif "datetime" in message or "time" in message:
                        self._log_skip(line_number, f"Invalid datetime format: {e}")
                    else:
                        self._log_skip(line_number, f"Invalid value: {e}")
                except TypeError as e:
                    self._log_skip(line_number, f"Data structure mismatch: {e}")
                except Exception as e:
                    logger.warning(f"{line_number} Unexpected exception ({type(e).__name__}): {e}")
                else:
                    self.jobs.append(job)

        if not self.jobs:
            logger.warning(f"No valid jobs found in {self.jobs_file}.")

    def add_job(self, job: MetriqGymJob) -> str:
        self.jobs.append(job)
        with open(self.jobs_file, "a") as file:
            file.write(job.serialize() + "\n")
        return job.id

    def get_jobs(self) -> list[MetriqGymJob]:
        return self.jobs

    def get_latest_job(self) -> MetriqGymJob:
        if not self.jobs:
            raise ValueError("No jobs available")
        return self.jobs[-1]

    def get_job(self, job_id: str) -> MetriqGymJob:
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise ValueError(f"Job with id {job_id} not found")

    def delete_job(self, job_id: str) -> None:
        self.jobs = [job for job in self.jobs if job.id != job_id]
        temp_file = f"{self.jobs_file}.tmp"
        try:
            with open(temp_file, "w") as file:
                for job in self.jobs:
                    file.write(job.serialize() + "\n")
            os.replace(temp_file, self.jobs_file)
            logger.info(f"Deleted job with id {job_id} from {self.jobs_file}")
        except Exception as e:
            logger.error(f"Failed to delete job with id {job_id}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
