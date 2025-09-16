import types

import metriq_gym.quantinuum.job as qjob


class DummyJobs:
    def __init__(self):
        self.cancel_called_with = None

    def cancel(self, ref):
        self.cancel_called_with = ref
        return None

    def results(self, _):
        return []


def test_cancel_calls_qnexus_jobs_cancel(monkeypatch):
    fake_qnx = types.SimpleNamespace()
    fake_qnx.jobs = DummyJobs()
    monkeypatch.setattr(qjob, "qnx", fake_qnx, raising=True)

    job_id = "abc-uuid"
    job = qjob.QuantinuumJob(job_id)

    assert job.cancel() is True
    assert fake_qnx.jobs.cancel_called_with == job_id

