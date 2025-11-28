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
    # Quantinuum always returns a list of counts, even for single circuit
    assert counts == [{"0": 1, "1": 3}]


def test_quantinuum_job_result_handles_batch_submissions(monkeypatch):
    """Test that batch submissions with multiple circuits return a list of counts.

    This test addresses the connectivity fix for benchmarks like BSEQ that require
    submitting multiple circuits to cover all qubit pairs.
    """
    # Build a fake jobs module with multiple results
    fake_qnx = types.SimpleNamespace()
    fake_qnx.jobs = FakeJobs(
        [
            FakeItem({(0, 0): 5, (1, 1): 3}),  # First circuit results
            FakeItem({(1, 0): 7, (0, 1): 1}),  # Second circuit results
            FakeItem({(1,): 4, (0,): 2}),  # Third circuit results
        ]
    )
    monkeypatch.setattr(qjob, "qnx", fake_qnx, raising=True)

    job = qjob.QuantinuumJob("job-uuid")
    res = job.result()
    assert isinstance(res.data, GateModelResultData)
    counts = res.data.measurement_counts

    # Verify it's a list with 3 entries
    assert isinstance(counts, list)
    assert len(counts) == 3

    # Verify each entry is normalized correctly
    assert counts[0] == {"00": 5, "11": 3}
    assert counts[1] == {"10": 7, "01": 1}
    assert counts[2] == {"0": 2, "1": 4}
