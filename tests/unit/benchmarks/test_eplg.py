"""Unit tests for EPLG benchmark."""

import argparse
from unittest.mock import MagicMock, patch

import rustworkx as rx

from metriq_gym.benchmarks.eplg import (
    EPLG,
    EPLGData,
    random_chain_from_graph,
)


def test_random_chain_from_graph_path():
    """Test random chain from path graph."""
    graph = rx.generators.path_graph(10)
    chain = random_chain_from_graph(graph, 5, seed=42)

    assert len(chain) == 5
    assert len(set(chain)) == 5
    # Verify it's a valid path
    for i in range(len(chain) - 1):
        assert graph.has_edge(chain[i], chain[i + 1])


def test_random_chain_from_graph_complete():
    """Test random chain from complete graph."""
    graph = rx.generators.complete_graph(10)
    chain = random_chain_from_graph(graph, 5, seed=42)

    assert len(chain) == 5
    assert len(set(chain)) == 5


def _make_eplg(decompose_clifford_ops: bool) -> EPLG:
    params = MagicMock()
    params.num_qubits_in_chain = 5
    params.two_qubit_gate = "cx"
    params.one_qubit_basis_gates = ["sx", "rz"]
    params.lengths = [4]
    params.num_samples = 1
    params.seed = 42
    params.shots = 100
    params.decompose_clifford_ops = decompose_clifford_ops
    return EPLG(argparse.Namespace(), params)


def test_dispatch_records_eplg_input_and_transpiled_gate_counts():
    benchmark = _make_eplg(decompose_clifford_ops=True)
    device = MagicMock()
    device.run.return_value.id = "job-eplg"
    graph = rx.generators.path_graph(10)

    with (
        patch("metriq_gym.benchmarks.eplg.connectivity_graph_for_gate", return_value=graph),
        patch("metriq_gym.benchmarks.eplg.connectivity_graph", return_value=graph),
    ):
        result = benchmark.dispatch_handler(device)

    assert isinstance(result, EPLGData)
    assert len(result.input_two_qubit_gate_counts) == len(result.transpiled_two_qubit_gate_counts)
    assert len(result.input_two_qubit_gate_counts) > 0
    assert sum(result.transpiled_two_qubit_gate_counts) >= sum(result.input_two_qubit_gate_counts)


def test_dispatch_mirrors_eplg_gate_counts_without_decomposition():
    benchmark = _make_eplg(decompose_clifford_ops=False)
    device = MagicMock()
    device.run.return_value.id = "job-eplg"
    graph = rx.generators.path_graph(10)

    with (
        patch("metriq_gym.benchmarks.eplg.connectivity_graph_for_gate", return_value=graph),
        patch("metriq_gym.benchmarks.eplg.connectivity_graph", return_value=graph),
    ):
        result = benchmark.dispatch_handler(device)

    assert result.input_two_qubit_gate_counts == result.transpiled_two_qubit_gate_counts
