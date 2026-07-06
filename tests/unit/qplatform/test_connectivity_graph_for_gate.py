"""Unit tests for connectivity_graph_for_gate."""

from unittest.mock import Mock

import pytest
import rustworkx as rx
from qbraid import QuantumDevice
from qbraid.runtime import QiskitBackend
from qiskit.circuit.library import CXGate, CZGate, ECRGate
from qiskit.transpiler import Target

from metriq_gym.qplatform.device import connectivity_graph_for_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_qiskit_device(target: Target) -> QiskitBackend:
    """Return a mocked QiskitBackend whose _backend.target is *target*."""
    device = Mock(spec=QiskitBackend)
    backend = Mock()
    backend.target = target
    device._backend = backend
    return device


def _make_linear_target(num_qubits: int, gate_cls=ECRGate, gate_name: str = "ecr") -> Target:
    """Create a Target with a single gate on a linear chain 0-1-\u2026-(n-1)."""
    target = Target(num_qubits=num_qubits)
    props = {(i, i + 1): None for i in range(num_qubits - 1)}
    target.add_instruction(gate_cls(), props, name=gate_name)
    return target


@pytest.fixture
def multi_gate_device():
    """Device with ecr on a full 5-qubit linear chain and cx on only two edges."""
    target = Target(num_qubits=5)
    target.add_instruction(
        ECRGate(),
        {(i, i + 1): None for i in range(4)},
        name="ecr",
    )
    target.add_instruction(
        CXGate(),
        {(0, 1): None, (2, 3): None},
        name="cx",
    )
    return _make_qiskit_device(target)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_returns_graph_for_known_gate():
    """Known gate in the target should return a PyGraph."""
    device = _make_qiskit_device(_make_linear_target(5))
    graph = connectivity_graph_for_gate(device, "ecr")
    assert isinstance(graph, rx.PyGraph)


def test_returns_none_for_unknown_gate():
    """Gate not present in the target should return None."""
    device = _make_qiskit_device(_make_linear_target(5))
    assert connectivity_graph_for_gate(device, "nonexistent") is None


def test_returns_none_for_unsupported_device_type():
    """The base singledispatch returns None for unknown device types."""
    device = Mock(spec=QuantumDevice)
    assert connectivity_graph_for_gate(device, "cx") is None


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_linear_chain_node_count():
    """Graph should contain all qubits in the target."""
    device = _make_qiskit_device(_make_linear_target(5))
    graph = connectivity_graph_for_gate(device, "ecr")
    assert graph.num_nodes() == 5


def test_linear_chain_edge_count():
    """Linear chain of n qubits has n-1 edges."""
    device = _make_qiskit_device(_make_linear_target(5))
    graph = connectivity_graph_for_gate(device, "ecr")
    assert graph.num_edges() == 4


def test_linear_chain_edges():
    """Every adjacent pair should be connected."""
    num_qubits = 5
    device = _make_qiskit_device(_make_linear_target(num_qubits))
    graph = connectivity_graph_for_gate(device, "ecr")
    edges = set(graph.edge_list())
    for i in range(num_qubits - 1):
        assert (i, i + 1) in edges or (i + 1, i) in edges


def test_graph_is_undirected():
    """Coupling maps are directed; the function must return an undirected PyGraph."""
    device = _make_qiskit_device(_make_linear_target(4))
    graph = connectivity_graph_for_gate(device, "ecr")
    assert isinstance(graph, rx.PyGraph)
    assert not isinstance(graph, rx.PyDiGraph)


# ---------------------------------------------------------------------------
# Per-gate filtering
# ---------------------------------------------------------------------------


def test_ecr_gets_full_chain(multi_gate_device):
    """ecr is defined on all 4 edges of the 5-qubit chain."""
    graph = connectivity_graph_for_gate(multi_gate_device, "ecr")
    assert graph.num_edges() == 4


def test_cx_gets_restricted_edges(multi_gate_device):
    """cx is only defined on 2 of the 4 edges."""
    graph = connectivity_graph_for_gate(multi_gate_device, "cx")
    assert graph.num_edges() == 2


def test_cx_edges_are_correct(multi_gate_device):
    """cx edges should be exactly (0,1) and (2,3)."""
    graph = connectivity_graph_for_gate(multi_gate_device, "cx")
    edges = set(graph.edge_list())
    for expected in [(0, 1), (2, 3)]:
        assert expected in edges or (expected[1], expected[0]) in edges


def test_missing_gate_among_others(multi_gate_device):
    """A gate not in the target returns None even when other gates exist."""
    assert connectivity_graph_for_gate(multi_gate_device, "cz") is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_edge():
    """Target with a single 2-qubit edge."""
    target = Target(num_qubits=2)
    target.add_instruction(CXGate(), {(0, 1): None}, name="cx")
    device = _make_qiskit_device(target)

    graph = connectivity_graph_for_gate(device, "cx")
    assert graph.num_nodes() == 2
    assert graph.num_edges() == 1


def test_disconnected_edges():
    """Gate defined on non-adjacent qubit pairs."""
    target = Target(num_qubits=6)
    target.add_instruction(CZGate(), {(0, 1): None, (4, 5): None}, name="cz")
    device = _make_qiskit_device(target)

    graph = connectivity_graph_for_gate(device, "cz")
    assert graph.num_edges() == 2
    edges = set(graph.edge_list())
    for expected in [(0, 1), (4, 5)]:
        assert expected in edges or (expected[1], expected[0]) in edges


def test_empty_target_returns_none():
    """An empty Target has no gates registered."""
    target = Target(num_qubits=3)
    device = _make_qiskit_device(target)
    assert connectivity_graph_for_gate(device, "cx") is None


@pytest.mark.parametrize("n", [2, 5, 10, 20])
def test_various_sizes(n):
    """Linear chain of size n should give n nodes and n-1 edges."""
    device = _make_qiskit_device(_make_linear_target(n))
    graph = connectivity_graph_for_gate(device, "ecr")
    assert graph.num_nodes() == n
    assert graph.num_edges() == n - 1
