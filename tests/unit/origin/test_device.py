import types

import pytest
from qiskit import QuantumCircuit

from metriq_gym.origin import device as origin_device
from metriq_gym.origin.device import OriginDevice


@pytest.fixture(autouse=True)
def stub_converter(monkeypatch):
    monkeypatch.setattr(
        origin_device,
        "convert_qasm_string_to_qprog",
        lambda qasm: f"converted({len(qasm)})",
        raising=False,
    )
    yield


class DummyBackend:
    def __init__(self):
        self.calls = []

    def run(self, *args):
        self.calls.append(args)
        return types.SimpleNamespace(job_id=lambda: "JOB123")


def _make_device(backend_name: str) -> tuple[OriginDevice, DummyBackend]:
    backend = DummyBackend()
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
