from datetime import datetime

import pytest

from metriq_gym.benchmarks.benchmark import BenchmarkResult
from metriq_gym.constants import JobType
from metriq_gym.exporters.dict_exporter import DictExporter
from metriq_gym.job_manager import MetriqGymJob
from metriq_gym.platform import canonical_device_name, canonical_provider_name


CEPHEUS_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"


@pytest.mark.parametrize("alias", ["aws", "AWS", "braket", "Braket"])
def test_aws_provider_aliases_share_canonical_identity(alias):
    assert canonical_provider_name(alias) == "aws"
    assert canonical_device_name(alias, CEPHEUS_ARN) == "rigetti_cepheus-1-108q"


def test_braket_job_and_export_use_canonical_platform_identity():
    job = MetriqGymJob(
        id="job-1",
        job_type=JobType.WIT,
        params={},
        data={},
        provider_name="braket",
        device_name=CEPHEUS_ARN,
        platform={"provider": "braket", "device": CEPHEUS_ARN},
        dispatch_time=datetime(2026, 7, 15),
    )

    assert job.provider_name == "aws"
    assert job.device_name == "rigetti_cepheus-1-108q"
    assert job.platform == {
        "provider": "aws",
        "device": "rigetti_cepheus-1-108q",
    }

    payload = DictExporter(job, BenchmarkResult()).export()
    assert payload["platform"] == {
        "provider": "aws",
        "device": "rigetti_cepheus-1-108q",
    }
