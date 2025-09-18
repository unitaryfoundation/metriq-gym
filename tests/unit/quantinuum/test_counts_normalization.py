import types


import metriq_gym.quantinuum.job as qjob
from qbraid.runtime import GateModelResultData


class FakeItem:
    def __init__(self, counts):
        self._counts = counts

    def download_result(self):
        class R:
            def __init__(self, c):
                self._c = c

            def get_counts(self):
                return self._c

        return R(self._counts)


class FakeJobs:
    def __init__(self, items):
        self._items = items

    def results(self, _):
        return self._items

    def wait_for(self, _):
        return None


def test_quantinuum_job_result_normalizes_to_single_bit(monkeypatch):
    # Build a fake jobs module compatible with qjob expectations
    fake_qnx = types.SimpleNamespace()
    fake_qnx.jobs = FakeJobs([FakeItem({(1,): 3, (0,): 1})])
    monkeypatch.setattr(qjob, "qnx", fake_qnx, raising=True)

    job = qjob.QuantinuumJob("job-uuid")
    res = job.result()
    assert isinstance(res.data, GateModelResultData)
    counts = res.data.measurement_counts
    assert counts == {"0": 1, "1": 3}
