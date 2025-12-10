import re
from datetime import datetime

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


def test_job_filename_structure():
    when = datetime(2024, 1, 2, 3, 4, 5)

    job = MetriqGymJob(
        id="job-1",
        job_type=JobType.QML_KERNEL,
        params={},
        data={},
        provider_name="local",
        device_name="aer_simulator",
        dispatch_time=when,
    )

    name = job_filename(job, rand_bytes=1)  # deterministic length
    assert re.match(
        r"2024-01-02_03-04-05_qml_kernel_[0-9a-f]{2}\.json",
        name,
    )


def test_suite_filename_structure():
    when = datetime(2024, 6, 7, 8, 9, 10)
    name = suite_filename("My Suite", when, rand_bytes=1)
    assert re.match(
        r"2024-06-07_08-09-10_my_suite_[0-9a-f]{2}\.json",
        name,
    )
