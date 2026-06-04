"""Unit tests for 2Q gate count tracking in BenchmarkData.

Tests the feature added in issue #715:
- input_two_qubit_gate_counts: one entry per input circuit
- transpiled_two_qubit_gate_counts: one entry per transpiled circuit
"""

import argparse
from unittest.mock import MagicMock, patch

import rustworkx as rx

from metriq_gym.benchmarks.benchmark import BenchmarkData


class TestBenchmarkDataGateCounts:
    """Test that BenchmarkData correctly stores 2Q gate count fields."""

    def test_base_benchmark_data_defaults_to_none(self):
        """New fields default to None when not provided."""
        data = BenchmarkData(provider_job_ids=["job-1"])
        assert data.input_two_qubit_gate_counts is None
        assert data.transpiled_two_qubit_gate_counts is None

    def test_base_benchmark_data_stores_gate_counts(self):
        """Can set gate count fields explicitly."""
        data = BenchmarkData(
            provider_job_ids=["job-1"],
            input_two_qubit_gate_counts=[1, 2, 3],
            transpiled_two_qubit_gate_counts=[2, 4, 6],
        )
        assert data.input_two_qubit_gate_counts == [1, 2, 3]
        assert data.transpiled_two_qubit_gate_counts == [2, 4, 6]

    def test_from_quantum_job_preserves_gate_counts(self):
        """from_quantum_job() passes through gate count kwargs."""
        mock_job = MagicMock()
        mock_job.id = "job-test"

        data = BenchmarkData.from_quantum_job(
            mock_job,
            input_two_qubit_gate_counts=[5, 10],
            transpiled_two_qubit_gate_counts=[7, 14],
        )
        assert data.provider_job_ids == ["job-test"]
        assert data.input_two_qubit_gate_counts == [5, 10]
        assert data.transpiled_two_qubit_gate_counts == [7, 14]


class TestCLOPSGateCounts:
    """Test CLOPS (non-transpiling) benchmark populates gate counts correctly."""

    def _linear_graph(self, n: int) -> rx.PyGraph:
        g = rx.PyGraph()
        g.add_nodes_from(range(n))
        for i in range(n - 1):
            g.add_edge(i, i + 1, None)
        return g

    def _make_params(self, **overrides):
        defaults = dict(
            benchmark_name="CLOPS",
            num_qubits=4,
            num_layers=2,
            num_circuits=3,
            shots=10,
            seed=42,
            two_qubit_gate="cz",
            mode="instantiated",
        )
        defaults.update(overrides)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def _make_clops(self, **param_overrides):
        from metriq_gym.benchmarks.clops import Clops

        args = argparse.Namespace()
        params = self._make_params(**param_overrides)
        return Clops(args, params)

    def test_dispatch_instantiated_sets_gate_counts(self):
        """CLOPS instantiated mode sets input and transpiled to same values."""
        from metriq_gym.benchmarks.clops import ClopsData

        clops = self._make_clops(mode="instantiated", use_session=False, num_circuits=3)
        device = MagicMock()
        device.profile.basis_gates = ["cz", "sx", "rz"]

        graph = self._linear_graph(6)
        mock_job = MagicMock()
        mock_job.id = "job-clops"
        device.run.return_value = mock_job

        with (
            patch(
                "metriq_gym.benchmarks.clops.connectivity_graph_for_gate",
                return_value=graph,
            ),
            patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph),
            patch(
                "metriq_gym.benchmarks.clops.pruned_connectivity_graph",
                return_value=graph,
            ),
        ):
            result = clops.dispatch_handler(device)

        assert isinstance(result, ClopsData)
        assert result.input_two_qubit_gate_counts is not None
        assert result.transpiled_two_qubit_gate_counts is not None
        # CLOPS doesn't transpile, so both should be identical
        assert result.input_two_qubit_gate_counts == result.transpiled_two_qubit_gate_counts
        # Should have one count per circuit
        assert len(result.input_two_qubit_gate_counts) == 3
        # All counts should be non-negative integers
        for count in result.input_two_qubit_gate_counts:
            assert isinstance(count, int)
            assert count >= 0

    def test_dispatch_parameterized_sets_gate_counts(self):
        """CLOPS parameterized mode sets gate counts on template."""
        from metriq_gym.benchmarks.clops import ClopsData
        from qbraid.runtime import QiskitBackend

        clops = self._make_clops(mode="parameterized", num_circuits=2)
        device = MagicMock(spec=QiskitBackend)
        device.profile.basis_gates = ["cz", "sx", "rz"]

        graph = self._linear_graph(6)
        mock_job = MagicMock()
        mock_job.id = "job-clops-param"
        clops._submit_ibm_with_options = MagicMock(return_value=mock_job)

        with (
            patch(
                "metriq_gym.benchmarks.clops.connectivity_graph_for_gate",
                return_value=graph,
            ),
            patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph),
            patch(
                "metriq_gym.benchmarks.clops.pruned_connectivity_graph",
                return_value=graph,
            ),
        ):
            result = clops.dispatch_handler(device)

        assert isinstance(result, ClopsData)
        assert result.input_two_qubit_gate_counts is not None
        assert result.transpiled_two_qubit_gate_counts is not None
        assert len(result.input_two_qubit_gate_counts) == 2
        # Parameterized sends same template for all circuits
        assert (
            result.input_two_qubit_gate_counts[0]
            == result.input_two_qubit_gate_counts[1]
        )

    def test_dispatch_twirled_sets_gate_counts(self):
        """CLOPS twirled mode sets gate counts on fixed template."""
        from metriq_gym.benchmarks.clops import ClopsData
        from qbraid.runtime import QiskitBackend

        clops = self._make_clops(mode="twirled", num_circuits=5)
        device = MagicMock(spec=QiskitBackend)
        device.profile.basis_gates = ["cz", "sx", "rz"]

        graph = self._linear_graph(6)
        mock_job = MagicMock()
        mock_job.id = "job-clops-twirled"
        clops._submit_ibm_with_options = MagicMock(return_value=mock_job)

        with (
            patch(
                "metriq_gym.benchmarks.clops.connectivity_graph_for_gate",
                return_value=graph,
            ),
            patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph),
            patch(
                "metriq_gym.benchmarks.clops.pruned_connectivity_graph",
                return_value=graph,
            ),
        ):
            result = clops.dispatch_handler(device)

        assert isinstance(result, ClopsData)
        assert len(result.input_two_qubit_gate_counts) == 5
        assert (
            result.input_two_qubit_gate_counts
            == result.transpiled_two_qubit_gate_counts
        )


class TestEPLGGateCounts:
    """Test EPLG (transpiling) benchmark populates both gate count fields."""

    def _make_eplg_params(self, **overrides):
        defaults = dict(
            benchmark_name="EPLG",
            num_qubits_in_chain=5,
            lengths=[4, 8],
            num_samples=2,
            shots=100,
            seed=42,
            two_qubit_gate="cx",
            one_qubit_basis_gates=["sx", "rz"],
            decompose_clifford_ops=True,
            constrain_rb_offset_b=False,
        )
        defaults.update(overrides)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def _make_eplg(self, **param_overrides):
        from metriq_gym.benchmarks.eplg import EPLG

        args = argparse.Namespace()
        params = self._make_eplg_params(**param_overrides)
        return EPLG(args, params)

    def test_dispatch_sets_both_gate_count_fields(self):
        """EPLG dispatch populates both input and transpiled gate counts."""
        from metriq_gym.benchmarks.eplg import EPLGData

        eplg = self._make_eplg(decompose_clifford_ops=True)
        device = MagicMock()
        device.num_qubits = 10
        mock_job = MagicMock()
        mock_job.id = "job-eplg"

        graph = rx.generators.path_graph(10)
        device.run.return_value = mock_job

        with (
            patch(
                "metriq_gym.benchmarks.eplg.connectivity_graph_for_gate",
                return_value=graph,
            ),
            patch("metriq_gym.benchmarks.eplg.connectivity_graph", return_value=graph),
        ):
            result = eplg.dispatch_handler(device)

        assert isinstance(result, EPLGData)
        assert result.input_two_qubit_gate_counts is not None
        assert result.transpiled_two_qubit_gate_counts is not None
        # Both should have same length (one per circuit)
        assert len(result.input_two_qubit_gate_counts) == len(
            result.transpiled_two_qubit_gate_counts
        )
        # Transpiled circuits should have >= total 2Q gates than input
        # (decomposition expands Clifford ops into native gates)
        assert sum(result.transpiled_two_qubit_gate_counts) >= sum(
            result.input_two_qubit_gate_counts
        )

    def test_dispatch_without_decompose_mirrors_counts(self):
        """EPLG without decompose_clifford_ops mirrors input=transpiled."""
        from metriq_gym.benchmarks.eplg import EPLGData

        eplg = self._make_eplg(decompose_clifford_ops=False)
        device = MagicMock()
        device.num_qubits = 10
        mock_job = MagicMock()
        mock_job.id = "job-eplg-no-decomp"

        graph = rx.generators.path_graph(10)
        device.run.return_value = mock_job

        with (
            patch(
                "metriq_gym.benchmarks.eplg.connectivity_graph_for_gate",
                return_value=graph,
            ),
            patch("metriq_gym.benchmarks.eplg.connectivity_graph", return_value=graph),
        ):
            result = eplg.dispatch_handler(device)

        assert isinstance(result, EPLGData)
        assert result.input_two_qubit_gate_counts == result.transpiled_two_qubit_gate_counts


class TestWITGateCounts:
    """Test WIT (non-transpiling) benchmark gate counts mirror input=transpiled."""

    def _make_wit_params(self, **overrides):
        defaults = dict(benchmark_name="WIT", num_qubits=6, shots=100)
        defaults.update(overrides)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def _make_wit(self, **param_overrides):
        from metriq_gym.benchmarks.wit import WIT

        args = argparse.Namespace()
        params = self._make_wit_params(**param_overrides)
        return WIT(args, params)

    def test_dispatch_sets_gate_counts_mirrored(self):
        """WIT sets input=transpiled since it doesn't transpile."""
        from metriq_gym.benchmarks.wit import WITData

        wit = self._make_wit()
        device = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-wit"
        device.run.return_value = mock_job

        result = wit.dispatch_handler(device)

        assert isinstance(result, WITData)
        assert result.input_two_qubit_gate_counts is not None
        assert result.transpiled_two_qubit_gate_counts is not None
        # WIT has a single circuit
        assert len(result.input_two_qubit_gate_counts) == 1
        assert result.input_two_qubit_gate_counts == result.transpiled_two_qubit_gate_counts
        # WIT uses cx, swap, rzz gates - should have non-zero 2Q gate count
        assert result.input_two_qubit_gate_counts[0] > 0


class TestQMLKernelGateCounts:
    """Test QMLKernel benchmark gate counts mirror input=transpiled."""

    def _make_qml_kernel_params(self, **overrides):
        defaults = dict(benchmark_name="QMLKernel", num_qubits=4, shots=100)
        defaults.update(overrides)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def _make_qml_kernel(self, **param_overrides):
        from metriq_gym.benchmarks.qml_kernel import QMLKernel

        args = argparse.Namespace()
        params = self._make_qml_kernel_params(**param_overrides)
        return QMLKernel(args, params)

    def test_dispatch_sets_gate_counts_mirrored(self):
        """QMLKernel sets input=transpiled since it doesn't transpile."""
        from metriq_gym.benchmarks.qml_kernel import QMLKernelData

        qml = self._make_qml_kernel()
        device = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-qml"
        device.run.return_value = mock_job

        result = qml.dispatch_handler(device)

        assert isinstance(result, QMLKernelData)
        assert result.input_two_qubit_gate_counts is not None
        assert result.transpiled_two_qubit_gate_counts is not None
        assert len(result.input_two_qubit_gate_counts) == 1
        assert result.input_two_qubit_gate_counts == result.transpiled_two_qubit_gate_counts
        # QML kernel uses cx gates
        assert result.input_two_qubit_gate_counts[0] > 0


class TestBSEQGateCounts:
    """Test BSEQ benchmark gate counts for circuit sets (multiple per color)."""

    def _make_bseq_params(self, **overrides):
        defaults = dict(benchmark_name="BSEQ", shots=100, max_colors=None)
        defaults.update(overrides)
        m = MagicMock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def _make_bseq(self, **param_overrides):
        from metriq_gym.benchmarks.bseq import BSEQ

        args = argparse.Namespace()
        params = self._make_bseq_params(**param_overrides)
        return BSEQ(args, params)

    def test_dispatch_sets_gate_counts_per_color(self):
        """BSEQ sets one gate count entry per color group."""
        from metriq_gym.benchmarks.bseq import BSEQData

        bseq = self._make_bseq()
        device = MagicMock()
        device.num_qubits = 5
        mock_job = MagicMock()
        mock_job.id = "job-bseq"
        device.run.return_value = mock_job

        graph = rx.generators.complete_graph(5)
        with patch(
            "metriq_gym.benchmarks.bseq.connectivity_graph", return_value=graph
        ):
            result = bseq.dispatch_handler(device)

        assert isinstance(result, BSEQData)
        assert result.input_two_qubit_gate_counts is not None
        assert result.transpiled_two_qubit_gate_counts is not None
        # BSEQ has one entry per color (4 basis measurements per color)
        # For complete graph on 5 qubits: 4 colors, so 4 circuit_sets
        # input = transpiled for BSEQ (no hardware transpilation)
        assert result.input_two_qubit_gate_counts == result.transpiled_two_qubit_gate_counts
        assert len(result.input_two_qubit_gate_counts) > 0
        # Each set should have CNOT gates (2 per edge in coloring)
        for count in result.input_two_qubit_gate_counts:
            assert count > 0
