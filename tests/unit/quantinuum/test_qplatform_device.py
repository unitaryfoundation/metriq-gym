"""
Unit tests for Quantinuum-specific handlers in qplatform.device:
- version(QuantinuumDevice) -> str
- connectivity_graph(QuantinuumDevice) -> rx.PyGraph

The device object is constructed directly (via qBraid's QuantinuumDevice
constructor) with a mocked backend_info, so these tests do not hit NEXUS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import pytest
import rustworkx as rx
from pytket import Circuit
from pytket.architecture import FullyConnected
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import TargetProfile
from qbraid.runtime.quantinuum import QuantinuumDevice

from metriq_gym.qplatform.device import version, connectivity_graph, normalized_metadata


@dataclass
class FakeBackendInfo:
    version: str
    architecture: Any


class SimpleArch:
    """Simple architecture stub with explicit nodes/edges."""

    def __init__(self, num_qubits: int, edges: Iterable[Tuple[int, int]]):
        self.nodes: List[int] = list(range(num_qubits))
        self.edges: List[Tuple[int, int]] = [(int(a), int(b)) for a, b in edges]


@pytest.fixture
def device_id() -> str:
    return "H1-1E"


def make_device(device_id: str, backend_info: Any) -> QuantinuumDevice:
    profile = TargetProfile(
        device_id=device_id,
        simulator="E" in device_id.upper(),
        experiment_type=ExperimentType.GATE_MODEL,
        num_qubits=len(backend_info.architecture.nodes),
        program_spec=ProgramSpec(Circuit, alias="pytket"),
        provider_name="quantinuum",
    )
    return QuantinuumDevice(profile=profile, backend_info=backend_info)


class TestVersionQuantinuum:
    def test_version_returns_backend_info_version(self, device_id: str):
        backend_info = FakeBackendInfo(version="2.3.4", architecture=FullyConnected(3))
        dev = make_device(device_id, backend_info)

        assert version(dev) == "2.3.4"


class TestConnectivityQuantinuum:
    def test_connectivity_fully_connected(self, device_id: str):
        n = 5
        backend_info = FakeBackendInfo(version="1.0.0", architecture=FullyConnected(n))
        dev = make_device(device_id, backend_info)

        g = connectivity_graph(dev)
        assert isinstance(g, rx.PyGraph)
        assert g.num_nodes() == n
        assert g.num_edges() == n * (n - 1) // 2

    def test_connectivity_sparse_explicit_edges(self, device_id: str):
        n = 6
        edges = [(0, 1), (1, 2), (3, 4)]
        backend_info = FakeBackendInfo(version="1.0.0", architecture=SimpleArch(n, edges))
        dev = make_device(device_id, backend_info)

        g = connectivity_graph(dev)
        assert isinstance(g, rx.PyGraph)
        assert g.num_nodes() == n
        assert g.num_edges() == len(edges)

        edge_list = {tuple(e) for e in g.edge_list()}
        for a, b in edges:
            assert (a, b) in edge_list or (b, a) in edge_list


class TestNumQubitsQuantinuum:
    def test_num_qubits_returns_architecture_node_count(self, device_id: str):
        n = 20
        backend_info = FakeBackendInfo(version="2.0.0", architecture=FullyConnected(n))
        dev = make_device(device_id, backend_info)

        assert dev.num_qubits == n

    def test_num_qubits_set_in_profile(self, device_id: str):
        n = 12
        backend_info = FakeBackendInfo(version="2.0.0", architecture=FullyConnected(n))
        dev = make_device(device_id, backend_info)

        assert dev.profile.num_qubits == n
        assert dev.num_qubits == n

    def test_normalized_metadata_includes_num_qubits(self, device_id: str):
        n = 32
        backend_info = FakeBackendInfo(version="3.1.0", architecture=FullyConnected(n))
        dev = make_device(device_id, backend_info)

        meta = normalized_metadata(dev)
        assert isinstance(meta, dict)
        assert meta.get("num_qubits") == n
        assert meta.get("version") == "3.1.0"
        # device_id contains 'E', so simulator should be True
        assert meta.get("simulator") is True
