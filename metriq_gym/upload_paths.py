"""Helpers for constructing deterministic upload paths and filenames."""

import re
import secrets
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metriq_gym.job_manager import MetriqGymJob


def minor_series_label(version: str) -> str:
    """
    Convert a semantic version into a v<major>.<minor> label.

    Examples:
        0.3.1      -> v0.3
        0.3.1.dev0 -> v0.3
        1.0        -> v1.0
        unknown    -> vunknown
    """
    match = re.match(r"(\d+)\.(\d+)", version)
    if match:
        return f"v{match.group(1)}.{match.group(2)}"
    return f"v{version}"


def path_component(value: str | None) -> str:
    """Normalize provider/device names for safe path segments."""
    cleaned = (value or "unknown").strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", cleaned)
    return cleaned.strip("_")


def default_upload_dir(version: str, provider: str, device: str) -> str:
    """Provider/device-aware default upload directory to avoid PR conflicts."""
    provider_part = path_component(provider)
    device_part = path_component(device)
    return f"metriq-gym/{minor_series_label(version)}/{provider_part}/{device_part}"


def job_filename(job: "MetriqGymJob", *, rand_bytes: int = 3) -> str:
    dispatch_time = job.dispatch_time or datetime.now()

    job_label = path_component(str(job.job_type.value))
    ts = dispatch_time.strftime("%Y-%m-%d_%H-%M-%S")
    rand = secrets.token_hex(rand_bytes)
    return f"{ts}_{job_label}_{rand}.json"


def suite_filename(
    suite_name: str | None, dispatch_time: datetime | None = None, *, rand_bytes: int = 3
) -> str:
    """Construct a filename for suite uploads using suite name and dispatch time."""
    suite_label = path_component(suite_name or "suite")
    ts = (dispatch_time or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    rand = secrets.token_hex(rand_bytes)
    return f"{ts}_{suite_label}_{rand}.json"
