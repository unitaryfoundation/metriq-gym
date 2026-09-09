import re
from datetime import datetime, timezone

import pytest

from metriq_gym.constants import JobType
from metriq_gym.job_manager import MetriqGymJob
from metriq_gym.upload_paths import (
    default_upload_dir,
    job_filename,
    minor_series_label,
    path_component,
    suite_filename,
)


def test_minor_series_label_parses_major_minor():
    assert minor_series_label("0.3.1") == "v0.3"
    assert minor_series_label("1.0") == "v1.0"
    assert minor_series_label("unknown") == "vunknown"


def test_path_component_sanitizes_and_defaults():
    assert path_component("My Provider") == "my_provider"
    assert path_component("aws/braket") == "aws_braket"
    assert path_component(None) == "unknown"


def test_default_upload_dir_builds_expected_path():
    path = default_upload_dir("0.4.2", "AWS/Braket", "Rigetti Aspen/M2")
    assert path == "metriq-gym/v0.4/aws_braket/rigetti_aspen_m2"


def test_default_upload_dir_canonicalizes_job_runtime_identifiers():
    job = MetriqGymJob(
        id="job-aws",
        job_type=JobType.WIT,
        params={},
        data={},
        provider_name="braket",
        device_name="arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
        dispatch_time=datetime(2026, 7, 15),
    )

    path = default_upload_dir("0.7.0", job.provider_name, job.device_name)
    assert path == "metriq-gym/v0.7/aws/rigetti_cepheus-1-108q"


def test_default_upload_dir_accepts_braket_alias_directly():
    path = default_upload_dir(
        "0.7.0",
        "braket",
        "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
    )
    assert path == "metriq-gym/v0.7/aws/rigetti_cepheus-1-108q"


def test_job_filename_structure():
    when = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    job = MetriqGymJob(
        id="job-1",
        job_type=JobType.QML_KERNEL,
        params={},
        data={},
        provider_name="local",
        device_name="aer_simulator",
        dispatch_time=when,
    )

    payload = {"results": {"score": {"value": 0.5, "uncertainty": None}}}
    name = job_filename(job, payload=payload)
    assert re.match(
        r"2024-01-02_03-04-05_qml_kernel_[0-9a-f]{8}\.json",
        name,
    )


def test_suite_filename_structure():
    when = datetime(2024, 6, 7, 8, 9, 10, tzinfo=timezone.utc)
    payload = [{"results": {"score": {"value": 1, "uncertainty": None}}}]
    name = suite_filename("My Suite", when, payload=payload)
    assert re.match(
        r"2024-06-07_08-09-10_my_suite_[0-9a-f]{8}\.json",
        name,
    )


@pytest.mark.parametrize(
    "dispatch_time, expected_prefix",
    [
        ("2026-09-09T14:34:31+00:00", "2026-09-09_14-34-31"),
        ("2026-09-09T16:34:31+02:00", "2026-09-09_14-34-31"),
        ("2026-09-09T23:34:31-05:00", "2026-09-10_04-34-31"),
        ("2026-09-09T16:34:31", "2026-09-09_14-34-31"),
        ("2026-01-09T16:34:31", "2026-01-09_15-34-31"),
    ],
)
def test_upload_filenames_use_utc(metriq_job, local_timezone, dispatch_time, expected_prefix):
    metriq_job.dispatch_time = datetime.fromisoformat(dispatch_time)

    assert job_filename(metriq_job) == f"{expected_prefix}_wit.json"
    assert (
        suite_filename("My Suite", metriq_job.dispatch_time) == f"{expected_prefix}_my_suite.json"
    )


def test_suite_filename_defaults_to_utc(local_timezone):
    before = datetime.now(timezone.utc).replace(microsecond=0)

    name = suite_filename("My Suite")

    after = datetime.now(timezone.utc)
    timestamp = datetime.strptime(name[:19], "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
    assert before <= timestamp <= after
