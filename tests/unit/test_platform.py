from datetime import datetime
import json

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


def test_unknown_provider_identifiers_are_preserved():
    assert canonical_provider_name("IBM") == "IBM"
    assert canonical_provider_name(" custom-provider ") == "custom-provider"


def test_braket_dataset_device_identity_is_region_independent():
    alternate_region_arn = CEPHEUS_ARN.replace("us-west-1", "us-east-1")

    assert canonical_device_name("aws", CEPHEUS_ARN) == canonical_device_name(
        "aws", alternate_region_arn
    )


def test_braket_job_preserves_runtime_identifiers_and_exports_canonical_platform():
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

    assert job.provider_name == "braket"
    assert job.device_name == CEPHEUS_ARN
    assert job.platform == {
        "provider": "aws",
        "device": "rigetti_cepheus-1-108q",
    }

    payload = DictExporter(job, BenchmarkResult()).export()
    assert payload["platform"] == {
        "provider": "aws",
        "device": "rigetti_cepheus-1-108q",
    }

    legacy_record = json.loads(job.serialize())
    legacy_record.pop("platform")
    restored_job = MetriqGymJob.deserialize(json.dumps(legacy_record))
    assert restored_job.provider_name == "braket"
    assert restored_job.device_name == CEPHEUS_ARN
    assert restored_job.platform == job.platform

    saved_record = json.loads(restored_job.serialize())
    assert saved_record["provider_name"] == "braket"
    assert saved_record["device_name"] == CEPHEUS_ARN
