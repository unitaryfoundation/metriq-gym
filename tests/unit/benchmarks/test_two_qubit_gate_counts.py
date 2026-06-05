"""Tests for two-qubit gate counting in benchmark data (issue #715).

Covers the counting helper, the new ``BenchmarkData`` fields (defaults,
round-trip, backwards compatibility), a non-transpiling benchmark (CLOPS), a
transpiling benchmark (EPLG), and the exporter surfacing the counts in JSON.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
from unittest.mock import MagicMock, patch

import rustworkx as rx
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import BenchmarkData
from metriq_gym.benchmarks.bseq import BSEQData
from metriq_gym.benchmarks.clops import Clops, ClopsData
from metriq_gym.circuits import two_qubit_gate_count, two_qubit_gate_counts
from metriq_gym.constants import JobType
from metriq_gym.exporters.dict_exporter import DictExporter
from metriq_gym.job_manager import MetriqGymJob


# ---------------------------------------------------------------------------
# Counting helper
# ---------------------------------------------------------------------------


def test_count_only_two_qubit_gates():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.barrier()
    qc.measure(range(3), range(3))

    assert two_qubit_gate_count(qc) == 3


def test_three_qubit_gate_not_counted():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)  # three-qubit, must not count

    assert two_qubit_gate_count(qc) == 1


def test_count_empty_circuit():
    assert two_qubit_gate_count(QuantumCircuit(2)) == 0


def test_parameterized_two_qubit_gate_counted():
    from qiskit.circuit import Parameter

    qc = QuantumCircuit(2)
    qc.rzz(Parameter("t"), 0, 1)

    assert two_qubit_gate_count(qc) == 1


def test_counts_accepts_single_circuit():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)

    assert two_qubit_gate_counts(qc) == [1]


def test_counts_preserves_order():
    a = QuantumCircuit(2)
    a.cx(0, 1)
    b = QuantumCircuit(3)
    b.cx(0, 1)
    b.cx(1, 2)
    c = QuantumCircuit(2)

    assert two_qubit_gate_counts([a, b, c]) == [1, 2, 0]


# ---------------------------------------------------------------------------
# BenchmarkData fields: defaults, round-trip, backwards compatibility
# ---------------------------------------------------------------------------


def test_benchmarkdata_defaults_to_empty_lists():
    data = BenchmarkData(provider_job_ids=["a"])

    assert data.input_two_qubit_gate_counts == []
    assert data.transpiled_two_qubit_gate_counts == []


def test_subclass_with_required_fields_accepts_counts_kwonly():
    # BSEQData adds required positional fields after the kw_only count fields;
    # this must not raise "non-default argument follows default".
    data = BSEQData(
        provider_job_ids=["a"],
        shots=10,
        num_qubits=3,
        input_two_qubit_gate_counts=[2, 2],
        transpiled_two_qubit_gate_counts=[2, 2],
    )

    assert data.input_two_qubit_gate_counts == [2, 2]
    assert data.transpiled_two_qubit_gate_counts == [2, 2]


def test_asdict_roundtrip_preserves_counts():
    data = ClopsData(
        provider_job_ids=["a"],
        input_two_qubit_gate_counts=[4, 4, 4],
        transpiled_two_qubit_gate_counts=[4, 4, 4],
    )
    restored = ClopsData(**asdict(data))

    assert restored == data


def test_old_record_without_counts_still_loads():
    # Records persisted before issue #715 lack the new keys; reconstruction must
    # default them to empty lists rather than fail.
    old_data = {"provider_job_ids": ["a"], "shots": 10, "num_qubits": 3}
    data = BSEQData(**old_data)

    assert data.input_two_qubit_gate_counts == []
    assert data.transpiled_two_qubit_gate_counts == []


# ---------------------------------------------------------------------------
# Non-transpiling benchmark (CLOPS): transpiled mirrors input
# ---------------------------------------------------------------------------


def _linear_graph(n: int) -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    for i in range(n - 1):
        g.add_edge(i, i + 1, None)
    return g


def _make_clops(**overrides) -> Clops:
    defaults = dict(
        benchmark_name="CLOPS",
        num_qubits=4,
        num_layers=2,
        num_circuits=3,
        shots=10,
        seed=42,
        two_qubit_gate="cz",
        mode="instantiated",
        use_session=False,
    )
    defaults.update(overrides)
    params = MagicMock()
    for k, v in defaults.items():
        setattr(params, k, v)
    return Clops(argparse.Namespace(), params)


def test_clops_dispatch_populates_matching_counts():
    clops = _make_clops()
    device = MagicMock()
    graph = _linear_graph(6)
    mock_job = MagicMock()
    mock_job.id = "job-123"
    device.run.return_value = mock_job

    with (
        patch("metriq_gym.benchmarks.clops.connectivity_graph_for_gate", return_value=graph),
        patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph),
        patch("metriq_gym.benchmarks.clops.pruned_connectivity_graph", return_value=graph),
    ):
        result = clops.dispatch_handler(device)

    submitted = device.run.call_args[0][0]
    expected = two_qubit_gate_counts(submitted)

    # One entry per submitted circuit, and transpiled mirrors input (no transpile step).
    assert result.input_two_qubit_gate_counts == expected
    assert result.transpiled_two_qubit_gate_counts == expected
    assert len(result.input_two_qubit_gate_counts) == 3
    assert all(count > 0 for count in result.input_two_qubit_gate_counts)


# ---------------------------------------------------------------------------
# Transpiling benchmark (EPLG): both counts populated from the right circuits
# ---------------------------------------------------------------------------


def _make_eplg(**overrides):
    from metriq_gym.benchmarks.eplg import EPLG

    defaults = dict(
        benchmark_name="EPLG",
        num_qubits_in_chain=4,
        two_qubit_gate="cx",
        one_qubit_basis_gates=["rz", "sx", "x"],
        lengths=[1, 2],
        num_samples=1,
        shots=10,
        seed=0,
        decompose_clifford_ops=True,
    )
    defaults.update(overrides)
    params = MagicMock()
    for k, v in defaults.items():
        setattr(params, k, v)
    return EPLG(argparse.Namespace(), params)


def test_eplg_dispatch_counts_input_and_transpiled_separately():
    eplg = _make_eplg()
    device = MagicMock()
    mock_job = MagicMock()
    mock_job.id = "eplg-job"
    device.run.return_value = mock_job

    input_circuits = [QuantumCircuit(2) for _ in range(2)]
    for qc in input_circuits:
        qc.cx(0, 1)  # one 2Q gate each before "transpilation"

    transpiled_circuits = [QuantumCircuit(2) for _ in range(2)]
    for qc in transpiled_circuits:
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.cx(0, 1)  # three 2Q gates each after "transpilation"

    build_return = (
        transpiled_circuits,
        input_circuits,
        [0, 1, 2, 3],  # qubit_chain
        [[(0, 1)], [(1, 2)]],  # two_disjoint_layers
        "cx",
        ["rz", "sx", "x"],
    )
    with patch.object(eplg, "_build_circuits", return_value=build_return):
        result = eplg.dispatch_handler(device)

    # device received the transpiled circuits
    assert device.run.call_args[0][0] is transpiled_circuits
    # input counts reflect the pre-transpile circuits, transpiled counts the submitted ones
    assert result.input_two_qubit_gate_counts == [1, 1]
    assert result.transpiled_two_qubit_gate_counts == [3, 3]


# ---------------------------------------------------------------------------
# Exporter surfaces the counts in the exported record
# ---------------------------------------------------------------------------


def _make_metriq_job(data: dict) -> MetriqGymJob:
    return MetriqGymJob(
        id="test-job",
        job_type=JobType.CLOPS,
        params={"benchmark_name": "CLOPS"},
        data=data,
        provider_name="provider",
        device_name="device",
        dispatch_time=datetime(2026, 6, 5, 12, 0, 0),
    )


def test_exporter_includes_circuit_stats():
    job = _make_metriq_job(
        {
            "provider_job_ids": ["qid"],
            "input_two_qubit_gate_counts": [4, 4, 4],
            "transpiled_two_qubit_gate_counts": [6, 6, 6],
        }
    )
    result = MagicMock()
    result.model_dump.return_value = {"clops_score": 1.0}

    record = DictExporter(job, result).export()

    assert record["circuit_stats"]["input_two_qubit_gate_counts"] == [4, 4, 4]
    assert record["circuit_stats"]["transpiled_two_qubit_gate_counts"] == [6, 6, 6]


def test_exporter_omits_circuit_stats_when_absent():
    job = _make_metriq_job({"provider_job_ids": ["qid"]})
    result = MagicMock()
    result.model_dump.return_value = {"clops_score": 1.0}

    record = DictExporter(job, result).export()

    assert "circuit_stats" not in record
