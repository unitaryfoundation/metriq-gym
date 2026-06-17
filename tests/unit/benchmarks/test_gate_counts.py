"""Tests for two-qubit gate-count tracking on benchmark data.

Covers:
  - the public counting helpers in ``resource_estimation``,
  - the keyword-only base-class fields (and that subclasses with required
    fields still construct),
  - a transpiling benchmark (EPLG) where input and transpiled counts differ,
  - a non-transpiling benchmark (CLOPS) where transpiled mirrors input,
  - the exporter surfacing the counts in the exported record.
"""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock, patch

import rustworkx as rx
from qbraid import QuantumJob
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import BenchmarkData, BenchmarkResult
from metriq_gym.benchmarks.clops import Clops, ClopsData
from metriq_gym.benchmarks.eplg import EPLG, EPLGData
from metriq_gym.constants import JobType
from metriq_gym.exporters.dict_exporter import DictExporter
from metriq_gym.job_manager import MetriqGymJob
from metriq_gym.resource_estimation import count_two_qubit_gates, two_qubit_gate_counts


def _circuit_with_2q_gates(n: int) -> QuantumCircuit:
    """A 2-qubit circuit carrying exactly ``n`` two-qubit gates."""
    qc = QuantumCircuit(2)
    for _ in range(n):
        qc.cx(0, 1)
    return qc


class TestCountingHelpers:
    def test_counts_only_two_qubit_gates(self):
        qc = QuantumCircuit(3)
        qc.h(0)  # 1q
        qc.cx(0, 1)  # 2q
        qc.cz(1, 2)  # 2q
        qc.ccx(0, 1, 2)  # 3q, not counted
        assert count_two_qubit_gates(qc) == 2

    def test_excludes_barriers_and_measurements(self):
        # measure_all() inserts a (2-wide) barrier plus measurements; neither is
        # a gate and must not inflate the two-qubit count.
        qc = _circuit_with_2q_gates(2)
        qc.measure_all()
        assert count_two_qubit_gates(qc) == 2

    def test_two_qubit_gate_counts_single_circuit(self):
        assert two_qubit_gate_counts(_circuit_with_2q_gates(3)) == [3]

    def test_two_qubit_gate_counts_flat_list(self):
        circuits = [_circuit_with_2q_gates(1), _circuit_with_2q_gates(2)]
        assert two_qubit_gate_counts(circuits) == [1, 2]

    def test_two_qubit_gate_counts_nested_in_order(self):
        # e.g. BSEQ's list-of-circuit-sets is flattened in submission order.
        nested = [
            [_circuit_with_2q_gates(1)],
            [_circuit_with_2q_gates(2), _circuit_with_2q_gates(3)],
        ]
        assert two_qubit_gate_counts(nested) == [1, 2, 3]


class TestBenchmarkDataFields:
    def test_defaults_are_none(self):
        data = BenchmarkData(provider_job_ids=["j"])
        assert data.input_two_qubit_gate_counts is None
        assert data.transpiled_two_qubit_gate_counts is None

    def test_keyword_only_does_not_break_subclass_ordering(self):
        # Regression guard: subclasses add their own *required* fields after the
        # base. The new fields are keyword-only precisely so this still works.
        @dataclass
        class CustomData(BenchmarkData):
            required_field: int

        data = CustomData(
            provider_job_ids=["j"],
            required_field=7,
            input_two_qubit_gate_counts=[1, 2],
            transpiled_two_qubit_gate_counts=[3, 4],
        )
        assert data.required_field == 7
        assert data.input_two_qubit_gate_counts == [1, 2]
        assert data.transpiled_two_qubit_gate_counts == [3, 4]


class TestEPLGGateCounts:
    """EPLG genuinely transpiles, so input and transpiled counts can differ."""

    @patch("metriq_gym.benchmarks.eplg.LayerFidelity")
    @patch("metriq_gym.benchmarks.eplg.random_chain_from_graph", return_value=[0, 1])
    @patch("metriq_gym.benchmarks.eplg.connectivity_graph_for_gate")
    def test_input_and_transpiled_counts_differ(
        self, mock_conn_for_gate, _mock_chain, mock_layer_fidelity
    ):
        graph = rx.PyGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edge(0, 1, None)
        mock_conn_for_gate.return_value = graph

        # Logical circuit has 1 two-qubit gate; the transpiled one has 3.
        logical = [_circuit_with_2q_gates(1)]
        transpiled = [_circuit_with_2q_gates(3)]
        lfexp = MagicMock()
        lfexp.circuits.return_value = logical
        lfexp._transpiled_circuits.return_value = transpiled
        mock_layer_fidelity.return_value = lfexp

        params = MagicMock()
        params.num_qubits_in_chain = 2
        params.two_qubit_gate = "cx"
        params.one_qubit_basis_gates = ["sx", "rz"]
        params.lengths = [1]
        params.num_samples = 1
        params.seed = 0
        params.shots = 100
        params.decompose_clifford_ops = True  # submit the transpiled circuits

        mock_job = MagicMock(spec=QuantumJob)
        mock_job.id = "eplg_job"
        device = MagicMock()
        device.run.return_value = mock_job

        result = EPLG(MagicMock(), params).dispatch_handler(device)

        assert isinstance(result, EPLGData)
        assert result.input_two_qubit_gate_counts == [1]
        assert result.transpiled_two_qubit_gate_counts == [3]

    @patch("metriq_gym.benchmarks.eplg.LayerFidelity")
    @patch("metriq_gym.benchmarks.eplg.random_chain_from_graph", return_value=[0, 1])
    @patch("metriq_gym.benchmarks.eplg.connectivity_graph_for_gate")
    def test_counts_match_when_not_decomposing(
        self, mock_conn_for_gate, _mock_chain, mock_layer_fidelity
    ):
        graph = rx.PyGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edge(0, 1, None)
        mock_conn_for_gate.return_value = graph

        logical = [_circuit_with_2q_gates(2)]
        lfexp = MagicMock()
        lfexp.circuits.return_value = logical
        lfexp._transpiled_circuits.return_value = [_circuit_with_2q_gates(9)]
        mock_layer_fidelity.return_value = lfexp

        params = MagicMock()
        params.num_qubits_in_chain = 2
        params.two_qubit_gate = "cx"
        params.one_qubit_basis_gates = ["sx", "rz"]
        params.lengths = [1]
        params.num_samples = 1
        params.seed = 0
        params.shots = 100
        params.decompose_clifford_ops = False  # submit the logical circuits

        mock_job = MagicMock(spec=QuantumJob)
        mock_job.id = "eplg_job"
        device = MagicMock()
        device.run.return_value = mock_job

        result = EPLG(MagicMock(), params).dispatch_handler(device)

        # Without local transpilation the submitted circuits are the logical ones.
        assert result.input_two_qubit_gate_counts == [2]
        assert result.transpiled_two_qubit_gate_counts == [2]


class TestClopsGateCounts:
    """CLOPS submits circuits directly, so transpiled mirrors input."""

    @patch("metriq_gym.benchmarks.clops.instantiate_circuits")
    @patch.object(Clops, "_build_template")
    def test_transpiled_mirrors_input(self, mock_build_template, mock_instantiate):
        mock_build_template.return_value = (MagicMock(), [])
        circuits = [_circuit_with_2q_gates(4), _circuit_with_2q_gates(4)]
        mock_instantiate.return_value = circuits

        params = MagicMock()
        params.mode = "instantiated"
        params.num_circuits = 2
        params.shots = 100
        params.seed = 0
        params.use_session = False

        mock_job = MagicMock(spec=QuantumJob)
        mock_job.id = "clops_job"
        device = MagicMock()
        device.run.return_value = mock_job

        result = Clops(MagicMock(), params).dispatch_handler(device)

        assert isinstance(result, ClopsData)
        assert result.input_two_qubit_gate_counts == [4, 4]
        assert result.transpiled_two_qubit_gate_counts == [4, 4]
        assert result.input_two_qubit_gate_counts == result.transpiled_two_qubit_gate_counts


class _DummyResult(BenchmarkResult):
    metric: float


class TestExporterSurfacesCounts:
    def _job(self, data: dict) -> MetriqGymJob:
        return MetriqGymJob(
            id="job",
            job_type=JobType.CLOPS,
            params={"shots": 10},
            data=data,
            provider_name="provider",
            device_name="device",
            dispatch_time=datetime.now(),
        )

    def test_counts_present_in_export(self):
        job = self._job(
            {
                "provider_job_ids": ["q"],
                "input_two_qubit_gate_counts": [4, 4],
                "transpiled_two_qubit_gate_counts": [4, 4],
            }
        )
        record = DictExporter(job, _DummyResult(metric=1.0)).export()
        assert record["circuit_metadata"] == {
            "input_two_qubit_gate_counts": [4, 4],
            "transpiled_two_qubit_gate_counts": [4, 4],
        }

    def test_no_circuit_metadata_when_absent(self):
        # Backward compatibility: records without the counts emit no new block.
        job = self._job({"provider_job_ids": ["q"]})
        record = DictExporter(job, _DummyResult(metric=1.0)).export()
        assert "circuit_metadata" not in record
