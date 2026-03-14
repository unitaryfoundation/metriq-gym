import types
from dataclasses import dataclass
from datetime import datetime

import metriq_gym.run as runmod
from qbraid.runtime import GateModelResultData, QuantumJob


class FakeQuantumJob(QuantumJob):
    def __init__(self, job_id: str):
        super().__init__(job_id, None)

    def status(self):
        from qbraid.runtime import JobStatus

        return JobStatus.COMPLETED

    def result(self):
        # Return a minimal GateModelResultData
        return types.SimpleNamespace(data=GateModelResultData(measurement_counts={"1": 2, "0": 0}))

    def cancel(self):
        return True


@dataclass
class FakeJob:
    id: str
    job_type: str
    params: dict
    data: dict
    provider_name: str
    device_name: str
    dispatch_time: datetime
    result_data: dict | None = None


def test_fetch_result_returns_handler_result_for_quantinuum(monkeypatch):
    # Build a fake metriq job with valid params; expect handler result when tasks completed
    mjob = FakeJob(
        id="m1",
        job_type="WIT",
        params={"benchmark_name": "WIT", "num_qubits": 6, "shots": 10},
        data={"provider_job_ids": ["p1"]},
        provider_name="quantinuum",
        device_name="H1-1LE",
        dispatch_time=datetime.now(),
        result_data=None,
    )

    # Stub load_job to return our fake job
    monkeypatch.setattr(runmod, "load_job", lambda *_args, **_kw: FakeQuantumJob("p1"))

    # Minimal handler plumbing
    class FakeHandler:
        def __init__(self, *_a, **_k):
            pass

        def poll_handler(self, _d, result_data, _jobs):
            # Return a simple Pydantic-like object
            return types.SimpleNamespace(model_dump=lambda: {"expectation_value": 0.2})

    # Monkeypatch handler resolution
    monkeypatch.setattr(runmod, "setup_benchmark", lambda *a: FakeHandler())
    monkeypatch.setattr(runmod, "setup_benchmark_result_class", lambda *_: types.SimpleNamespace)

    @dataclass
    class FakeJobData:
        provider_job_ids: list[str]

    monkeypatch.setattr(runmod, "setup_job_data_class", lambda *_: FakeJobData)
    # Job manager with no-op update
    mgr = types.SimpleNamespace(update_job=lambda *_: None)

    # Create args with include_raw=False to avoid serialization issues with mocks
    args = types.SimpleNamespace(include_raw=False, no_cache=False)
    fetch_output = runmod.fetch_result(mjob, args, mgr)
    assert fetch_output.result.model_dump()["expectation_value"] == 0.2
    assert fetch_output.from_cache is False
