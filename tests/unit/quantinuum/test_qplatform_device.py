"""
Unit tests for Quantinuum-specific handlers in qplatform.device:
- version(QuantinuumDevice) -> str
- connectivity_graph(QuantinuumDevice) -> rx.PyGraph

We mock qnexus.devices.get_all() to avoid network calls and to control
the backend_info.version and backend_info.architecture used by the
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import pytest
import rustworkx as rx
import pandas as pd
import qnexus as qnx
from pytket.architecture import FullyConnected

from metriq_gym.quantinuum.device import QuantinuumDevice
from metriq_gym.qplatform.device import version, connectivity_graph


@dataclass
class FakeBackendInfo:
    version: str
    architecture: Any


class FakeDevicesResult:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def df(self) -> pd.DataFrame:  # noqa: D401 - simple passthrough
        """Return the DataFrame of devices."""
        return self._df


class SimpleArch:
    """Simple architecture stub with explicit nodes/edges."""

    def __init__(self, num_qubits: int, edges: Iterable[Tuple[int, int]]):
        self.nodes: List[int] = list(range(num_qubits))
        # normalize to tuples and ensure list
        self.edges: List[Tuple[int, int]] = [(int(a), int(b)) for a, b in edges]


@pytest.fixture
def device_id() -> str:
    return "H1-1E"


def make_device(device_id: str) -> QuantinuumDevice:
    # provider is unused by the tested functions; a simple object suffices
    return QuantinuumDevice(provider=object(), device_id=device_id)


def patch_qnexus(monkeypatch: pytest.MonkeyPatch, device_id: str, backend_info: Any) -> None:
    df = pd.DataFrame(
        [
            {
                "device_name": device_id,
                "backend_info": backend_info,
            }
        ]
    )

    def fake_get_all(issuers=None):  # signature-compatible enough for tests
        return FakeDevicesResult(df)

    monkeypatch.setattr(qnx.devices, "get_all", fake_get_all)


class TestVersionQuantinuum:
    def test_version_returns_backend_info_version(self, monkeypatch: pytest.MonkeyPatch, device_id: str):
        backend_info = FakeBackendInfo(version="2.3.4", architecture=FullyConnected(3))
        patch_qnexus(monkeypatch, device_id, backend_info)
        dev = make_device(device_id)

        assert version(dev) == "2.3.4"


class TestConnectivityQuantinuum:
    def test_connectivity_fully_connected(self, monkeypatch: pytest.MonkeyPatch, device_id: str):
        n = 5
        backend_info = FakeBackendInfo(version="1.0.0", architecture=FullyConnected(n))
        patch_qnexus(monkeypatch, device_id, backend_info)
        dev = make_device(device_id)

        g = connectivity_graph(dev)
        assert isinstance(g, rx.PyGraph)
        assert g.num_nodes() == n
        # complete graph edges
        assert g.num_edges() == n * (n - 1) // 2

    def test_connectivity_sparse_explicit_edges(self, monkeypatch: pytest.MonkeyPatch, device_id: str):
        n = 6
        edges = [(0, 1), (1, 2), (3, 4)]  # deliberately sparse
        backend_info = FakeBackendInfo(version="1.0.0", architecture=SimpleArch(n, edges))
        patch_qnexus(monkeypatch, device_id, backend_info)
        dev = make_device(device_id)

        g = connectivity_graph(dev)
        assert isinstance(g, rx.PyGraph)
        assert g.num_nodes() == n
        assert g.num_edges() == len(edges)

        # rustworkx edge_list returns a list of (u, v)
        edge_list = set(tuple(e) for e in g.edge_list())
        for a, b in edges:
            assert (a, b) in edge_list or (b, a) in edge_list
