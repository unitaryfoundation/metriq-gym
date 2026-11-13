"""Partial submission recovery through local SDKs with mocked provider calls."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from braket.circuits import Circuit
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import AzureQuantumDevice, BraketDevice, QiskitBackend, TargetProfile
from qbraid.runtime.origin import OriginDevice
from qbraid.runtime.quantinuum import QuantinuumDevice
from qiskit import QuantumCircuit

from metriq_gym.local.device import LocalAerDevice
from metriq_gym.qplatform.submission import (
    record_submission,
    run_circuits,
    submit_circuits,
    track_submissions,
)


def _remote_job(name):
    return SimpleNamespace(id=name)


def _remote_batch(name):
    return SimpleNamespace(tasks=[_remote_job(name)])


def _provider_device(provider, monkeypatch):
    """Use real qBraid submission code and mock only the remote SDK boundary."""
    if provider == "azure":
        profile = TargetProfile(
            device_id="azure-test",
            simulator=True,
            num_qubits=2,
            experiment_type=ExperimentType.GATE_MODEL,
            program_spec=ProgramSpec(QuantumCircuit),
        )
        workspace = Mock()
        device = AzureQuantumDevice(profile, workspace=workspace)
        remote = workspace.get_targets.return_value.submit
        result = _remote_job
        circuits = [QuantumCircuit(2) for _ in range(3)]
    else:
        aws_device = Mock()
        monkeypatch.setattr("qbraid.runtime.aws.device.AwsDevice", lambda **_: aws_device)
        profile = TargetProfile(
            device_id="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            simulator=True,
            num_qubits=34,
            experiment_type=ExperimentType.GATE_MODEL,
            program_spec=ProgramSpec(Circuit, alias="braket"),
            provider_name="Amazon Braket",
        )
        device = BraketDevice(profile)
        circuits = [Circuit().h(0).cnot(0, 1) for _ in range(3)]
        if provider == "braket-partial":
            for circuit in circuits:
                circuit.partial_measurement_qubits = [1]
            remote = aws_device.run
            result = _remote_job
        else:
            remote = aws_device.run_batch
            result = _remote_batch
    device.set_options(transpile=False, transform=False, validate=False, prepare=False)
    return device, remote, result, circuits


@pytest.mark.parametrize("provider", ["azure", "braket", "braket-partial"])
@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_later_provider_failure_preserves_prior_ids(provider, accepted, monkeypatch):
    device, remote, result, circuits = _provider_device(provider, monkeypatch)
    ids = [f"accepted-{i}" for i in range(accepted)]
    remote.side_effect = [result(job_id) for job_id in ids] + [RuntimeError("submission rejected")]
    recorded = []

    with track_submissions(recorded.extend), pytest.raises(RuntimeError, match="rejected"):
        run_circuits(device, circuits, shots=100)

    assert recorded == ids
    assert remote.call_count == accepted + 1


@pytest.mark.parametrize("provider", ["azure", "braket", "braket-partial"])
@pytest.mark.parametrize("batch", [False, True])
def test_real_sdk_success_preserves_order_and_return_shape(provider, batch, monkeypatch):
    device, remote, result, circuits = _provider_device(provider, monkeypatch)
    selected = circuits if batch else circuits[0]
    expected_ids = [f"accepted-{i}" for i in range(3 if batch else 1)]
    remote.side_effect = [result(job_id) for job_id in expected_ids]
    recorded = []

    with track_submissions(recorded.extend):
        jobs = run_circuits(device, selected, shots=123)

    assert isinstance(jobs, list) is batch
    assert [job.id for job in (jobs if batch else [jobs])] == expected_ids
    assert recorded == expected_ids
    assert all(c.kwargs["shots"] == 123 for c in remote.call_args_list)


@pytest.mark.parametrize("device_type", [AzureQuantumDevice, BraketDevice, OriginDevice])
def test_all_preparation_finishes_before_any_submission(device_type):
    device = Mock(spec=device_type)
    device.profile = SimpleNamespace(simulator=True)
    device.apply_runtime_profile.side_effect = ["prepared", ValueError("invalid second circuit")]

    with pytest.raises(ValueError, match="invalid second"):
        run_circuits(device, ["first", "second"], shots=100)

    device.submit.assert_not_called()
    device.run.assert_not_called()


def test_checkpoint_failure_stops_before_next_provider_submission(monkeypatch):
    device, remote, result, circuits = _provider_device("azure", monkeypatch)
    remote.side_effect = [result("accepted"), result("must-not-submit")]
    callback = Mock(side_effect=OSError("checkpoint failed"))

    with track_submissions(callback), pytest.raises(OSError, match="checkpoint failed"):
        run_circuits(device, circuits, shots=100)

    remote.assert_called_once()
    callback.assert_called_once_with(["accepted"])


def test_recording_context_is_restored_after_nested_failure():
    outer, inner = [], []
    job = SimpleNamespace(id="accepted")

    with track_submissions(outer.extend):
        assert record_submission(job) is job
        with pytest.raises(RuntimeError), track_submissions(inner.extend):
            record_submission([job])
            raise RuntimeError("inner dispatch failed")
        record_submission(job)
    record_submission(job)

    assert outer == ["accepted", "accepted"]
    assert inner == ["accepted"]


@pytest.mark.parametrize(
    "device_type", [QiskitBackend, QuantinuumDevice, LocalAerDevice, OriginDevice]
)
def test_native_batch_provider_run_is_preserved(device_type):
    device = Mock(spec=device_type)
    device.profile = SimpleNamespace(simulator=False)
    device.run.return_value = SimpleNamespace(id="batch")
    circuits = [object(), object()]
    recorded = []

    with track_submissions(recorded.extend):
        result = run_circuits(device, circuits, "positional", shots=100)

    assert result is device.run.return_value
    device.run.assert_called_once_with(circuits, "positional", shots=100)
    device.apply_runtime_profile.assert_not_called()
    device.submit.assert_not_called()
    assert recorded == ["batch"]


def test_prepared_native_batch_is_recorded_without_runtime_conversion():
    device = Mock(spec=QiskitBackend)
    device.submit.return_value = SimpleNamespace(id="native-batch")
    circuits = [object(), object()]
    recorded = []

    with track_submissions(recorded.extend):
        result = submit_circuits(device, circuits, shots=100)

    assert result is device.submit.return_value
    device.submit.assert_called_once_with(circuits, shots=100)
    device.apply_runtime_profile.assert_not_called()
    assert recorded == ["native-batch"]


def test_origin_simulator_fanout_and_expanded_conversion_preserve_order():
    device = Mock(spec=OriginDevice)
    device.profile = SimpleNamespace(simulator=True)
    device.apply_runtime_profile.side_effect = [["first-a", "first-b"], "second"]
    device.submit.side_effect = [SimpleNamespace(id=str(i)) for i in range(3)]
    recorded = []

    with track_submissions(recorded.extend):
        jobs = run_circuits(device, ["first", "second"], shots=100)

    assert [job.id for job in jobs] == recorded == ["0", "1", "2"]
    assert device.submit.call_args_list == [
        call("first-a", shots=100),
        call("first-b", shots=100),
        call("second", shots=100),
    ]


def test_empty_batch_does_not_submit_or_checkpoint():
    device = Mock(spec=BraketDevice)
    callback = Mock()

    with track_submissions(callback):
        assert run_circuits(device, [], shots=100) == []

    device.submit.assert_not_called()
    callback.assert_not_called()
