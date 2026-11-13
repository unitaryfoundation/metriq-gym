"""Benchmark submission boundaries preserve accepted jobs when later work fails."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import rustworkx as rx

from metriq_gym.benchmarks import bseq
from metriq_gym.benchmarks.clops import Clops
from metriq_gym.qplatform.submission import track_submissions


def test_bseq_tracks_first_batch_before_second_batch_fails(monkeypatch):
    graph = rx.generators.path_graph(3)
    monkeypatch.setattr(bseq, "connectivity_graph", lambda device: graph)
    benchmark = bseq.BSEQ(Namespace(), SimpleNamespace(shots=100, max_colors=None))
    device = Mock()
    device.num_qubits = 3
    device.run.side_effect = [SimpleNamespace(id="accepted-batch"), RuntimeError("rejected")]
    checkpoints = []

    with track_submissions(checkpoints.append):
        with pytest.raises(RuntimeError, match="rejected"):
            benchmark.dispatch_handler(device)

    assert checkpoints == [["accepted-batch"]]
    assert device.run.call_count == 2


def test_clops_tracks_ibm_job_before_session_cleanup_fails(monkeypatch):
    sampler = Mock()
    sampler.run.return_value.job_id.return_value = "accepted-ibm-job"
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(side_effect=RuntimeError("session cleanup failed"))
    monkeypatch.setattr("qiskit_ibm_runtime.Session", Mock(return_value=session))
    monkeypatch.setattr("qiskit_ibm_runtime.SamplerV2", Mock(return_value=sampler))
    monkeypatch.setattr(
        "qbraid.runtime.ibm.job.QiskitJob",
        lambda job_id, **kwargs: SimpleNamespace(id=job_id),
    )
    benchmark = Clops(Namespace(), SimpleNamespace())
    checkpoints = []

    with track_submissions(checkpoints.append):
        with pytest.raises(RuntimeError, match="session cleanup failed"):
            benchmark._submit_ibm_with_options(Mock(), [], shots=100, use_session=True)

    assert checkpoints == [["accepted-ibm-job"]]
    sampler.run.assert_called_once_with([], shots=100)
