from datetime import datetime

import pytest

from metriq_gym.constants import JobType
from metriq_gym.job_manager import MetriqGymJob


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
