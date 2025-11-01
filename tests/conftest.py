from datetime import datetime

import pytest

from metriq_gym.constants import JobType
from metriq_gym.job_manager import MetriqGymJob


@pytest.fixture
def metriq_job() -> MetriqGymJob:
    return MetriqGymJob(
        id="test-job",
        job_type=JobType.WIT,
        params={
            "benchmark_name": "WIT",
            "shots": 10,
            "n_qubits_per_side": 3,
            "message_size": 1,
            "insert_message_method": "reset",
            "interaction_coupling_strength": 1.5707963267948966,
            "x_rotation_transverse_angle": 0.7853981633974483,
            "zz_rotation_angle": 0.7853981633974483,
            "z_rotation_angles": [0.0283397, 0.00519953, 0.0316079],
            "time_steps": 3,
            "num_qubits": 6,
        },
        data={"provider_job_ids": ["qid"]},
        provider_name="provider",
        device_name="device",
        dispatch_time=datetime.now(),
    )
