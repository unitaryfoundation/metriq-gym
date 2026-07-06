import types


import metriq_gym.quantinuum.job as qjob
from qbraid.runtime import GateModelResultData
from pytket.circuit import BasisOrder


class FakeItem:
    def __init__(self, counts):
        # Assumes count keys are least signficant bit first
        self._counts = counts

    def download_result(self):
        class R:
            def __init__(self, c):
                self._c = c

            def get_counts(self, basis: BasisOrder = BasisOrder.ilo):
                # Mock the behavior of the BasisOrder argument in pytket
                if basis == BasisOrder.dlo:
                    # Convert keys to most significant bit first
                    return {tuple(reversed(k)): v for k, v in self._c.items()}
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


def test_quantinuum_job_result_handles_batch_submissions(monkeypatch):
    fake_qnx = types.SimpleNamespace()
    fake_qnx.jobs = FakeJobs(
        [
            FakeItem({(0, 0): 5, (1, 1): 3}),
            FakeItem({(0, 1): 7, (1, 0): 2}),
            FakeItem({(1, 1): 4, (0, 0): 6}),
        ]
    )
    monkeypatch.setattr(qjob, "qnx", fake_qnx, raising=True)

    job = qjob.QuantinuumJob("batch-job-uuid")
    res = job.result()
    assert isinstance(res.data, GateModelResultData)
    counts = res.data.measurement_counts

    assert isinstance(counts, list)
    assert len(counts) == 3
    assert counts[0] == {"00": 5, "11": 3}
    assert counts[1] == {"10": 7, "01": 2}
    assert counts[2] == {"11": 4, "00": 6}
