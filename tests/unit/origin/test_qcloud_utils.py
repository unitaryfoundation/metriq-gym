import types

import pytest

from metriq_gym.origin import qcloud_utils


def _stub_qcloud_module(captured):
    module = types.SimpleNamespace()

    class DummyService:
        def __init__(self, api_key):
            captured.append(("service", api_key))

    class DummyJob:
        def __init__(self, job_id):
            captured.append(("job", job_id))
            self.job_id = job_id

    class DummyOptions:
        def __init__(self):
            captured.append(("options", None))

    class DummyJobStatus:
        FINISHED = "FINISHED"
        WAITING = "WAITING"
        COMPUTING = "COMPUTING"
        FAILED = "FAILED"

    module.QCloudService = DummyService
    module.QCloudJob = DummyJob
    module.QCloudOptions = DummyOptions
    module.JobStatus = DummyJobStatus
    return module


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    qcloud_utils._SERVICE_CACHE.clear()
    monkeypatch.delenv("ORIGIN_API_KEY", raising=False)
    # Reset module reference to the real qcloud module after each test
    monkeypatch.delattr(qcloud_utils, "qcloud_module", raising=False)


def test_get_service_uses_env_and_caches(monkeypatch):
    events = []
    module = _stub_qcloud_module(events)
    monkeypatch.setenv("ORIGIN_API_KEY", "secret-key")
    monkeypatch.setattr(qcloud_utils, "qcloud_module", module, raising=False)

    service1 = qcloud_utils.get_service()
    service2 = qcloud_utils.get_service()

    assert service1 is service2
    assert events.count(("service", "secret-key")) == 1


def test_get_qcloud_job_initializes_service(monkeypatch):
    events = []
    module = _stub_qcloud_module(events)
    monkeypatch.setenv("ORIGIN_API_KEY", "another-key")
    monkeypatch.setattr(qcloud_utils, "qcloud_module", module, raising=False)

    job = qcloud_utils.get_qcloud_job("JOB123")

    assert job.job_id == "JOB123"
    assert ("service", "another-key") in events
    assert ("job", "JOB123") in events
    # Service should be created before job instantiation
    assert events.index(("service", "another-key")) < events.index(("job", "JOB123"))


def test_resolve_api_key_missing(monkeypatch):
    monkeypatch.delenv("ORIGIN_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        qcloud_utils.resolve_api_key(None)
