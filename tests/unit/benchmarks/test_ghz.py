import numpy as np
import pytest
import rustworkx as rx
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.ghz import (
    GHZResult,
    _bfs_edges,
    _select_flag_qubits,
    build_ghz_circuits,
    estimate_fidelity_dfe,
    estimate_fidelity_oscillation,
    post_select_results,
)
from metriq_gym.benchmarks.benchmark import BenchmarkScore


class TestBfsEdges:
    def test_linear_graph(self):
        graph = rx.generators.path_graph(5)
        edges = _bfs_edges(graph, root=0, num_qubits=5)
        assert len(edges) == 4
        # BFS from 0 on a path: 0-1, 1-2, 2-3, 3-4
        assert edges == [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_complete_graph(self):
        graph = rx.generators.complete_graph(6)
        edges = _bfs_edges(graph, root=0, num_qubits=6)
        assert len(edges) == 5
        # On complete graph, BFS from 0 gives star: 0->1, 0->2, ..., 0->5
        for ctrl, _targ in edges:
            assert ctrl == 0

    def test_partial_qubits(self):
        graph = rx.generators.path_graph(10)
        edges = _bfs_edges(graph, root=0, num_qubits=4)
        assert len(edges) == 3
        # Should only use first 4 nodes
        all_nodes = {0}
        for c, t in edges:
            all_nodes.add(c)
            all_nodes.add(t)
        assert len(all_nodes) == 4


class TestSelectFlagQubits:
    def test_selects_neighbors_outside_data(self):
        # Path graph: 0-1-2-3-4
        graph = rx.generators.path_graph(5)
        data = {0, 1, 2}
        flags = _select_flag_qubits(graph, data, num_flags=1)
        assert len(flags) == 1
        assert flags[0] == 3  # neighbor of 2, not in data

    def test_no_flags_requested(self):
        graph = rx.generators.path_graph(5)
        flags = _select_flag_qubits(graph, {0, 1, 2}, num_flags=0)
        assert flags == []

    def test_complete_graph_many_flags(self):
        graph = rx.generators.complete_graph(8)
        data = {0, 1, 2, 3}
        flags = _select_flag_qubits(graph, data, num_flags=3)
        assert len(flags) == 3
        assert all(f not in data for f in flags)


class TestBuildGhzCircuits:
    def test_dfe_returns_two_circuits(self):
        graph = rx.generators.complete_graph(4)
        circuits, data_qubits, flag_qubits = build_ghz_circuits(graph, num_qubits=4, method="dfe")
        assert len(circuits) == 2
        assert all(isinstance(c, QuantumCircuit) for c in circuits)
        assert len(data_qubits) == 4
        assert flag_qubits == []

    def test_parity_oscillation_returns_correct_count(self):
        graph = rx.generators.complete_graph(4)
        phases = np.linspace(0, 2 * np.pi, 10, endpoint=False).tolist()
        circuits, data_qubits, flag_qubits = build_ghz_circuits(
            graph, num_qubits=4, method="parity_oscillation", phases=phases
        )
        # 1 z-basis + 10 oscillation circuits
        assert len(circuits) == 11
        assert len(data_qubits) == 4

    def test_with_flag_qubits(self):
        graph = rx.generators.path_graph(6)
        circuits, data_qubits, flag_qubits = build_ghz_circuits(
            graph, num_qubits=4, method="dfe", num_flag_qubits=1
        )
        assert len(circuits) == 2
        assert len(data_qubits) == 4
        assert len(flag_qubits) == 1
        assert flag_qubits[0] not in data_qubits

    def test_unknown_method_raises(self):
        graph = rx.generators.complete_graph(4)
        with pytest.raises(ValueError, match="Unknown verification method"):
            build_ghz_circuits(graph, num_qubits=4, method="invalid")

    def test_parity_oscillation_no_phases_raises(self):
        graph = rx.generators.complete_graph(4)
        with pytest.raises(ValueError, match="phases required"):
            build_ghz_circuits(graph, num_qubits=4, method="parity_oscillation")


class TestPostSelectResults:
    def test_no_flags(self):
        counts = {"00": 50, "11": 50}
        result = post_select_results(counts, num_flag_qubits=0)
        assert result == counts

    def test_filters_by_flag_qubits(self):
        # Format: [flags][data] — flags on the left
        counts = {
            "000": 40,  # flag=0, data=00 -> keep
            "011": 30,  # flag=0, data=11 -> keep
            "100": 20,  # flag=1, data=00 -> discard
            "111": 10,  # flag=1, data=11 -> discard
        }
        result = post_select_results(counts, num_flag_qubits=1)
        assert result == {"00": 40, "11": 30}

    def test_all_flagged_returns_empty(self):
        counts = {"100": 50, "111": 50}
        result = post_select_results(counts, num_flag_qubits=1)
        assert result == {}


class TestEstimateFidelityDfe:
    def test_perfect_ghz(self):
        n = 3
        z_counts = {"000": 500, "111": 500}
        x_counts = {
            "000": 250,
            "011": 250,
            "101": 250,
            "110": 250,
        }  # all even parity
        pop, coh, p_err, c_err = estimate_fidelity_dfe(z_counts, x_counts, n, num_flag_qubits=0)
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(1.0)

    def test_maximally_mixed(self):
        n = 2
        # Uniform distribution over all 4 bitstrings
        z_counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        x_counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        pop, coh, _, _ = estimate_fidelity_dfe(z_counts, x_counts, n, num_flag_qubits=0)
        assert pop == pytest.approx(0.5)
        assert coh == pytest.approx(0.0)

    def test_empty_counts_returns_zero(self):
        pop, coh, p_err, c_err = estimate_fidelity_dfe({}, {}, 3, 0)
        assert pop == 0.0
        assert coh == 0.0


class TestEstimateFidelityOscillation:
    def test_perfect_oscillation(self):
        n = 4
        phases = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        # Perfect oscillation: cos(4*phi)
        osc_counts_list = []
        for phi in phases:
            parity = np.cos(n * phi)
            # Convert parity to counts: even_frac = (1 + parity) / 2
            total = 1000
            even = int(total * (1 + parity) / 2)
            odd = total - even
            osc_counts_list.append({"00": even, "01": odd})

        z_counts = {"0000": 500, "1111": 500}
        pop, coh, _, _ = estimate_fidelity_oscillation(
            z_counts, osc_counts_list, phases.tolist(), n, num_flag_qubits=0
        )
        assert pop == pytest.approx(1.0)
        assert coh == pytest.approx(1.0, abs=0.05)

    def test_empty_z_counts(self):
        pop, coh, _, _ = estimate_fidelity_oscillation({}, [], [], 3, 0)
        assert pop == 0.0
        assert coh == 0.0


class TestGHZResult:
    def test_compute_score(self):
        result = GHZResult(
            population=BenchmarkScore(value=0.9, uncertainty=0.01),
            coherence=BenchmarkScore(value=0.8, uncertainty=0.02),
            fidelity=BenchmarkScore(value=0.85, uncertainty=0.01),
        )
        assert result.compute_score() == result.fidelity

    def test_values_dict(self):
        result = GHZResult(
            population=BenchmarkScore(value=0.9, uncertainty=0.01),
            coherence=BenchmarkScore(value=0.8, uncertainty=0.02),
            fidelity=BenchmarkScore(value=0.85, uncertainty=0.01),
        )
        vals = result.values
        assert "population" in vals
        assert "coherence" in vals
        assert "fidelity" in vals
