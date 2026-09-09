from datetime import datetime
import time

import pytest

from metriq_gym.constants import JobType
from metriq_gym.job_manager import MetriqGymJob


@pytest.fixture
def local_timezone(monkeypatch, request):
    """Run with a non-UTC system timezone, restoring it even if the test fails."""
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the system timezone requires time.tzset")
    try:
        with monkeypatch.context() as env:
            env.setenv("TZ", getattr(request, "param", "Europe/Madrid"))
            time.tzset()
            yield
    finally:
        time.tzset()


@pytest.fixture
def metriq_job() -> MetriqGymJob:
    return MetriqGymJob(
        id="test-job",
        job_type=JobType.WIT,
        params={"benchmark_name": "WIT", "shots": 10},
        data={"provider_job_ids": ["qid"]},
        provider_name="provider",
        device_name="device",
        dispatch_time=datetime.now(),
    )
