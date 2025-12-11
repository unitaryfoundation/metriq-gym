"""Local persistence and helpers for tracking dispatched metriq-gym jobs."""

from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
import shutil
from metriq_gym._version import __version__
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
    platform: dict[str, Any] | None = None
    suite_id: str | None = None
    suite_name: str | None = None
    # No suite weights in this PR; reserved for future use
    app_version: str | None = __version__
    result_data: dict[str, Any] | None = None
    runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        """Keep platform and provider/device fields in sync on initialization.

        - If platform is missing, populate from provider_name/device_name.
        - If platform exists but lacks keys, backfill them from provider/device fields.
        """
        plat = self.platform or {}
        if not plat:
            plat = {"provider": self.provider_name, "device": self.device_name}
        else:
            if "provider" not in plat:
                plat["provider"] = self.provider_name
            if "device" not in plat:
                plat["device"] = self.device_name
        self.platform = plat

    def to_table_row(self, show_suite_id: bool) -> list[str | None]:
        return (
            [
                self.suite_id,
            ]
            if show_suite_id
            else []
        ) + [
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
        job_dict["job_type"] = JobType(job_dict["job_type"])
        job_dict["dispatch_time"] = datetime.fromisoformat(job_dict["dispatch_time"])
        # Backwards/forwards compatibility for platform vs. provider/device fields
        platform = job_dict.get("platform")
        if not platform and "provider_name" in job_dict and "device_name" in job_dict:
            job_dict["platform"] = {
                "provider": job_dict["provider_name"],
                "device": job_dict["device_name"],
            }
        if "provider_name" not in job_dict or "device_name" not in job_dict:
            plat = job_dict.get("platform", {})
            job_dict.setdefault("provider_name", plat.get("provider"))
            job_dict.setdefault("device_name", plat.get("device"))
        # Drop any unknown fields to keep older/newer records loadable
        allowed_keys = {f.name for f in fields(MetriqGymJob)}
        job_dict = {k: v for k, v in job_dict.items() if k in allowed_keys}
        return MetriqGymJob(**job_dict)

    def __str__(self) -> str:
        rows: list[list[str | None]] = [
            ["suite_id", self.suite_id],
            ["suite_name", self.suite_name],
            ["id", self.id],
            ["job_type", self.job_type.value],
            ["params", pprint.pformat(self.params)],
            ["provider_name", self.provider_name],
            ["device_name", self.device_name],
            ["platform", pprint.pformat(self.platform)],
            ["provider_job_ids", pprint.pformat(self.data["provider_job_ids"])],
            ["dispatch_time", self.dispatch_time.isoformat()],
            ["app_version", self.app_version],
            ["runtime_seconds", str(self.runtime_seconds)],
            ["result_data", pprint.pformat(self.result_data)],
        ]
        return tabulate(rows, tablefmt="fancy_grid")


# TODO: https://github.com/unitaryfoundation/metriq-gym/issues/51
class JobManager:
    jobs: list[MetriqGymJob]
    jobs_file: Path

    def __init__(self) -> None:
        self.jobs_file = get_data_db_path()
        # Track original lines (parsed jobs and raw skipped) to preserve order on rewrite.
        # Each entry is (line_number, kind, payload) where kind is "job" or "raw".
        self._line_entries: list[tuple[int, str, Any]] = []
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

        if not self.jobs_file.exists():
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
                    self._line_entries.append((line_number, "job", job))
                    continue
                # Keep skipped lines so later rewrites don't drop user data and preserve order
                self._line_entries.append((line_number, "raw", stripped_line))

        if not self.jobs:
            logger.warning(f"No valid jobs found in {self.jobs_file}.")

    def add_job(self, job: MetriqGymJob) -> str:
        self.jobs.append(job)
        max_line = max((ln for ln, _, _ in self._line_entries), default=0)
        self._line_entries.append((max_line + 1, "job", job))
        # Append safely without rewriting existing records (minimize data-loss risk)
        try:
            with open(self.jobs_file, "a") as file:
                file.write(job.serialize() + "\n")
        except Exception as e:
            logger.error(f"Failed to append job {job.id} to {self.jobs_file}: {e}")
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

    def get_jobs_by_suite_id(self, suite_id: str) -> list[MetriqGymJob]:
        return [job for job in self.jobs if job.suite_id == suite_id]

    def delete_job(self, job_id: str) -> None:
        self.jobs = [job for job in self.jobs if job.id != job_id]
        self._line_entries = [
            (ln, kind, payload)
            for (ln, kind, payload) in self._line_entries
            if not (kind == "job" and isinstance(payload, MetriqGymJob) and payload.id == job_id)
        ]
        self._rewrite_jobs_file()
        logger.info(f"Deleted job with id {job_id} from {self.jobs_file}")

    def update_job(self, updated_job: MetriqGymJob) -> None:
        """Persist updated job information to disk."""
        for idx, job in enumerate(self.jobs):
            if job.id == updated_job.id:
                self.jobs[idx] = updated_job
                break
        else:
            raise ValueError(f"Cannot update job: job with id {updated_job.id} not found")

        new_entries: list[tuple[int, str, Any]] = []
        for ln, kind, payload in self._line_entries:
            if kind == "job" and isinstance(payload, MetriqGymJob) and payload.id == updated_job.id:
                new_entries.append((ln, "job", updated_job))
            else:
                new_entries.append((ln, kind, payload))
        self._line_entries = new_entries
        self._rewrite_jobs_file()

    def _rewrite_jobs_file(self) -> None:
        """Rewrite the jobs file preserving original line order (parsed + skipped)."""
        backup_path = self._backup_jobs_file()
        temp_file = f"{self.jobs_file}.tmp"
        try:
            with open(temp_file, "w") as file:
                for ln, kind, payload in sorted(self._line_entries, key=lambda x: x[0]):
                    if kind == "job":
                        file.write(payload.serialize() + "\n")
                    else:
                        file.write(str(payload).strip() + "\n")
            os.replace(temp_file, self.jobs_file)
        except Exception as e:
            logger.error(f"Failed to rewrite jobs file {self.jobs_file}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if backup_path:
                logger.error(f"Original file preserved at {backup_path}")

    def _backup_jobs_file(self) -> Path | None:
        """Create a timestamped backup of the current jobs file before rewrite."""
        if not self.jobs_file.exists():
            return None
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_path = Path(f"{self.jobs_file}.{ts}.bak")
        try:
            shutil.copy2(self.jobs_file, backup_path)
            return backup_path
        except Exception as e:
            logger.error(f"Failed to back up {self.jobs_file}: {e}")
            return None
