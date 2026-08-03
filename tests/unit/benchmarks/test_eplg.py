"""Unit tests for EPLG benchmark."""

from unittest.mock import MagicMock, patch

import pytest
import rustworkx as rx

from metriq_gym.benchmarks.eplg import (
    EPLG,
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


def test_eplg_warns_when_device_gate_support_cannot_be_confirmed():
    """EPLG tells users how to adapt its device-dependent gate parameters."""
    graph = rx.generators.path_graph(2)
    layer_fidelity = MagicMock()
    layer_fidelity.circuits.return_value = []

    params = MagicMock()
    params.num_qubits_in_chain = 2
    params.two_qubit_gate = "cz"
    params.one_qubit_basis_gates = ["rz", "rx", "x"]
    params.lengths = [2]
    params.num_samples = 1
    params.seed = 12345
    params.decompose_clifford_ops = False

    with (
        patch(
            "metriq_gym.benchmarks.eplg.connectivity_graph_for_gate",
            return_value=None,
        ),
        patch("metriq_gym.benchmarks.eplg.connectivity_graph", return_value=graph),
        patch(
            "metriq_gym.benchmarks.eplg.random_chain_from_graph",
            return_value=[0, 1],
        ),
        patch(
            "metriq_gym.benchmarks.eplg.LayerFidelity",
            return_value=layer_fidelity,
        ),
        pytest.warns(
            RuntimeWarning,
            match="provider/device-specific.*change them to a native universal gate set",
        ),
    ):
        EPLG(MagicMock(), params)._build_circuits(MagicMock())
