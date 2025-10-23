import types

import pytest
from qiskit import QuantumCircuit

from metriq_gym.origin import device as origin_device
from metriq_gym.origin.device import OriginDevice, get_origin_connectivity


@pytest.fixture(autouse=True)
def stub_converter(monkeypatch):
    monkeypatch.setattr(
        origin_device,
        "convert_qasm_string_to_qprog",
        lambda qasm: f"converted({len(qasm)})",
        raising=False,
    )
    yield


class DummyChipInfo:
    def __init__(self, *, high=None, available=None, edges=None):
        self._high = [] if high is None else list(high)
        self._available = [] if available is None else list(available)
        self._edges = [] if edges is None else list(edges)

    def high_frequency_qubits(self):
        return self._high

    def available_qubits(self):
        return self._available

    def get_chip_topology(self):
        return self._edges

    def double_qubits_info(self):
        return []

    def qubits_num(self):
        return 102

    def get_basic_gates(self):
        return ["x", "cx"]


class DummyBackend:
    def __init__(self, chip_info=None):
        self.calls = []
        self._chip_info = chip_info or DummyChipInfo(high=[0, 1, 2, 3], edges=[(0, 1)])

    def run(self, *args):
        self.calls.append(args)
        return types.SimpleNamespace(job_id=lambda: "JOB123")

    def chip_info(self):
        return self._chip_info


def _make_device(backend_name: str, *, chip_info=None) -> tuple[OriginDevice, DummyBackend]:
    backend = DummyBackend(chip_info=chip_info)
    provider = types.SimpleNamespace()
    device = OriginDevice(
        provider=provider,
        device_id=backend_name,
        backend=backend,
        backend_name=backend_name,
    )
    return device, backend


def test_simulator_submission_omits_options(monkeypatch):
    device, backend = _make_device("full_amplitude")
    monkeypatch.setattr(
        origin_device,
        "get_qcloud_options",
        lambda: pytest.fail("get_qcloud_options should not be called for simulators"),
        raising=False,
    )

    circuit = QuantumCircuit(1)
    device.submit(circuit, shots=123)

    assert len(backend.calls) == 1
    args = backend.calls[0]
    assert len(args) == 2
    assert args[1] == 123


def test_simulator_profile_reports_known_qubits():
    device, _ = _make_device("full_amplitude")

    assert device.profile.num_qubits == 35


def test_qpu_submission_uses_options(monkeypatch):
    device, backend = _make_device("WK_C102_400")
    sentinel_options = object()
    monkeypatch.setattr(
        origin_device,
        "get_qcloud_options",
        lambda: sentinel_options,
        raising=False,
    )

    circuit = QuantumCircuit(1)
    device.submit(circuit, shots=55)

    assert len(backend.calls) == 1
    args = backend.calls[0]
    assert len(args) == 3
    assert args[1] == 55
    assert args[2] is sentinel_options


def test_get_origin_connectivity_prefers_high_frequency_qubits():
    chip = DummyChipInfo(high=[5, 1, 3], available=[0, 1, 2, 3, 4], edges=[(0, 1), (1, 2)])
    device, _ = _make_device("WK_C102_400", chip_info=chip)

    active, edges = get_origin_connectivity(device)

    assert active == [1, 3, 5]
    assert edges == [(0, 1), (1, 2)]


def test_get_origin_connectivity_falls_back_to_edges():
    chip = DummyChipInfo(high=[], available=[], edges=[(8, 9), (9, 10)])
    device, _ = _make_device("WK_C102_400", chip_info=chip)

    active, edges = get_origin_connectivity(device)

    assert active == [8, 9, 10]
    assert edges == [(8, 9), (9, 10)]
