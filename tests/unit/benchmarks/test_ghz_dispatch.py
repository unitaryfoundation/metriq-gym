from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import QiskitBackend, TargetProfile
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from metriq_gym.benchmarks.ghz import GHZBenchmark
from metriq_gym.local.device import LocalAerDevice
from metriq_gym.schema_validator import validate_and_create_model


def _benchmark(**overrides):
    params = validate_and_create_model(
        {
            "benchmark_name": "GHZ",
            "num_qubits": 3,
            "num_stabilizers": 4,
            "shots": 100,
            "seed": 42,
            **overrides,
        }
    )
    return GHZBenchmark(Namespace(), params)


def _ibm_device(backend):
    profile = TargetProfile(
        device_id=backend.name,
        simulator=True,
        local=True,
        experiment_type=ExperimentType.GATE_MODEL,
        num_qubits=backend.num_qubits,
        program_spec=ProgramSpec(QuantumCircuit),
    )
    service = SimpleNamespace(backend=lambda *args, **kwargs: backend)
    return QiskitBackend(profile, service=service)


def _measured_qubits(circuit):
    return [
        circuit.find_bit(instruction.qubits[0]).index
        for instruction in circuit.data
        if instruction.operation.name == "measure"
    ]


def _preparation(circuit):
    instructions = []
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            break
        instructions.append(
            (
                instruction.operation.name,
                instruction.operation.params,
                [circuit.find_bit(qubit).index for qubit in instruction.qubits],
            )
        )
    return instructions


@pytest.mark.parametrize("sparse", [False, True], ids=["fake-manila", "sparse-layout"])
def test_ibm_dispatch_keeps_one_physical_preparation_for_all_settings(monkeypatch, sparse):
    if sparse:
        backend = GenericBackendV2(
            5,
            coupling_map=[[0, 4], [4, 0], [4, 2], [2, 4], [2, 1], [1, 2], [1, 3], [3, 1]],
            seed=42,
        )
        expected_qubits = [0, 2, 4]
    else:
        backend = FakeManilaV2()
        expected_qubits = [0, 1, 2]
    device = _ibm_device(backend)
    submit = Mock(return_value=SimpleNamespace(id="ghz-job"))
    monkeypatch.setattr(device, "submit", submit)
    transform = Mock(side_effect=AssertionError("Default transform would change the layout"))
    monkeypatch.setattr(device, "transform", transform)

    data = _benchmark().dispatch_handler(device)

    submitted = submit.call_args.args[0]
    assert len(submitted) == 5
    assert all(_measured_qubits(circuit) == expected_qubits for circuit in submitted)
    assert all(_preparation(circuit) == _preparation(submitted[0]) for circuit in submitted)
    assert data.provider_job_ids == ["ghz-job"]
    assert data.input_two_qubit_gate_counts == [2] * 5
    assert data.transpiled_two_qubit_gate_counts == [2] * 5
    transform.assert_not_called()
    assert device._options.get("transform") is True


def test_size_search_preserves_each_sizes_layout(monkeypatch):
    device = _ibm_device(FakeManilaV2())
    submit = Mock(return_value=SimpleNamespace(id="ghz-job"))
    monkeypatch.setattr(device, "submit", submit)

    data = _benchmark(size_search=True, min_qubits=2, method="dfe").dispatch_handler(device)

    assert data.search_sizes == [5, 2]
    assert data.search_circuit_counts == [5, 5]
    submitted = submit.call_args.args[0]
    for circuits, expected_qubits in [(submitted[:5], list(range(5))), (submitted[5:], [0, 1])]:
        assert all(_measured_qubits(circuit) == expected_qubits for circuit in circuits)
        assert all(_preparation(circuit) == _preparation(circuits[0]) for circuit in circuits)


@pytest.mark.parametrize("original_transform", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_dispatch_restores_runtime_options(monkeypatch, original_transform, fails):
    device = _ibm_device(FakeManilaV2())
    device.set_options(transform=original_transform)
    submit = Mock(return_value=SimpleNamespace(id="ghz-job"))
    if fails:
        submit.side_effect = RuntimeError("Submission failed")
    monkeypatch.setattr(device, "submit", submit)

    if fails:
        with pytest.raises(RuntimeError, match="Submission failed"):
            _benchmark().dispatch_handler(device)
    else:
        _benchmark().dispatch_handler(device)

    assert device._options.get("transform") is original_transform


def test_local_dispatch_preserves_layout_through_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("MGYM_LOCAL_SIMULATOR_CACHE_DIR", str(tmp_path))
    backend = AerSimulator.from_backend(FakeManilaV2())
    device = LocalAerDevice(provider=Mock(), backend=backend)
    run = Mock()
    run.return_value.result.return_value.get_counts.return_value = [{"000": 100}] * 5
    monkeypatch.setattr(backend, "run", run)

    _benchmark().dispatch_handler(device)

    submitted = run.call_args.args[0]
    assert all(_measured_qubits(circuit) == [0, 1, 2] for circuit in submitted)
    assert all(_preparation(circuit) == _preparation(submitted[0]) for circuit in submitted)
    assert device._options.get("transform") is True
