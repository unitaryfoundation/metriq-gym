"""IBM usage estimates use local circuits and timing metadata only."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Delay, Measure, Parameter, Reset
from qiskit.circuit.library import XGate
from qiskit.transpiler import InstructionProperties, Target

from metriq_gym.qplatform.ibm_estimation import estimate_ibm_usage
from metriq_gym.resource_estimation import CircuitBatch


@pytest.fixture
def device():
    target = Target(num_qubits=3, dt=1e-9)
    for operation, durations in [
        (XGate(), [2e-6, 4e-6, 6e-6]),
        (Measure(), [5e-6, 5e-6, 5e-6]),
        (Reset(), [7e-6, 8e-6, 100e-6]),
    ]:
        target.add_instruction(
            operation,
            {
                (qubit,): InstructionProperties(duration=value)
                for qubit, value in enumerate(durations)
            },
        )
    target.add_instruction(Delay(Parameter("duration")), {(q,): None for q in range(3)})
    return SimpleNamespace(
        _backend=SimpleNamespace(target=target, default_rep_delay=250e-6),
        transform=Mock(side_effect=lambda circuit: circuit),
        run=Mock(side_effect=AssertionError("must not run hardware")),
        submit=Mock(side_effect=AssertionError("must not submit jobs")),
    )


def parallel_circuit():
    circuit = QuantumCircuit(3, 2)
    circuit.x(0)
    circuit.x(1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def test_estimate_parallel_instructions_and_used_qubit_initialization(device):
    circuit = parallel_circuit()
    before = circuit.copy()

    result = estimate_ibm_usage([CircuitBatch([circuit], shots=1000)], device)

    # The parallel x gates take 4 us, measurements 5 us, and initialization 8 us.
    # The unused third qubit's slow reset must not inflate the result.
    assert result == pytest.approx(2 + 1000 * (9e-6 + 8e-6 + 250e-6))
    assert circuit == before
    assert device.transform.call_args.args[0] is not circuit
    device.run.assert_not_called()
    device.submit.assert_not_called()


def test_reuses_circuit_duration_across_batches_with_different_shots(device):
    circuit = parallel_circuit()
    batches = [
        CircuitBatch([circuit, circuit], shots=100),
        CircuitBatch([], shots=100),
        CircuitBatch([circuit], shots=200),
    ]

    assert estimate_ibm_usage(iter(batches), device) == pytest.approx(4 + 400 * 267e-6)
    assert device.transform.call_count == 1


def test_physical_circuit_skips_transpilation(device):
    circuit = QuantumCircuit(3, 1)
    circuit.x(1)
    circuit.measure(1, 0)

    result = estimate_ibm_usage([CircuitBatch([circuit], shots=10, transpile=False)], device)

    assert result == pytest.approx(2 + 10 * 267e-6)
    device.transform.assert_not_called()


@pytest.mark.parametrize("legacy_duration", [False, True])
@pytest.mark.parametrize(
    "delay_before_x,delay_dt",
    [
        pytest.param(False, 100, id="measurement-alignment"),
        pytest.param(True, 100, id="pulse-alignment"),
        pytest.param(False, 112, id="already-aligned"),
    ],
)
def test_physical_circuit_respects_hardware_alignment(
    device,
    monkeypatch,
    legacy_duration,
    delay_before_x,
    delay_dt,
):
    target = device._backend.target
    target.acquire_alignment = 16
    target.pulse_alignment = 16
    for name in ("x", "measure"):
        target.update_instruction_properties(name, (1,), InstructionProperties(duration=160e-9))
    if legacy_duration:
        monkeypatch.setattr(QuantumCircuit, "estimate_duration", None)

    # Use physical qubit 1: remapping to qubit 0 would use different durations.
    circuit = QuantumCircuit(3, 1)
    if delay_before_x:
        circuit.delay(delay_dt, 1, unit="dt")
    circuit.x(1)
    if not delay_before_x:
        circuit.delay(delay_dt, 1, unit="dt")
    circuit.measure(1, 0)
    before = circuit.copy()

    result = estimate_ibm_usage([CircuitBatch([circuit], shots=1000, transpile=False)], device)

    # A 100 dt delay puts the next instruction off the 16 dt boundary; it must
    # become 112 dt. An already aligned 112 dt delay must not gain more padding.
    # 160 dt + 112 dt + 160 dt = 432 ns, plus qubit 1's 8 us initialization.
    assert result == pytest.approx(2 + 1000 * (432e-9 + 8e-6 + 250e-6), rel=0, abs=1e-12)
    assert circuit == before
    device.transform.assert_not_called()
    device.run.assert_not_called()
    device.submit.assert_not_called()


def test_uses_transformed_circuit_and_preserves_input(device):
    circuit = QuantumCircuit(3, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    before = circuit.copy()

    def transform(copy):
        copy.x(1)
        return copy

    device.transform.side_effect = transform
    result = estimate_ibm_usage([CircuitBatch([circuit], shots=10)], device)

    assert result == pytest.approx(2 + 10 * (7e-6 + 8e-6 + 250e-6))
    assert circuit == before


def test_cache_distinguishes_transformed_and_already_physical_circuits(device):
    circuit = parallel_circuit()
    batches = [
        CircuitBatch([circuit], shots=1, transpile=False),
        CircuitBatch([circuit], shots=1, transpile=True),
    ]

    assert estimate_ibm_usage(batches, device) == pytest.approx(4 + 2 * 267e-6)
    device.transform.assert_called_once()


def test_qiskit_14_duration_fallback(device, monkeypatch):
    monkeypatch.setattr(QuantumCircuit, "estimate_duration", None)

    result = estimate_ibm_usage([CircuitBatch([parallel_circuit()], shots=100)], device)

    assert result == pytest.approx(2 + 100 * 267e-6)


def test_missing_device():
    with pytest.raises(ValueError, match="requires --device"):
        estimate_ibm_usage([], None)


def test_missing_target():
    with pytest.raises(ValueError, match="target timing metadata"):
        estimate_ibm_usage([], SimpleNamespace())


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1])
def test_invalid_repetition_delay(device, value):
    device._backend.default_rep_delay = value

    with pytest.raises(ValueError, match="default_rep_delay"):
        estimate_ibm_usage([CircuitBatch([parallel_circuit()], shots=10)], device)


@pytest.mark.parametrize("name", ["x", "reset", "measure"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1])
def test_invalid_instruction_metadata(device, name, value):
    device._backend.target.update_instruction_properties(
        name, (0,), InstructionProperties(duration=value)
    )

    with pytest.raises(ValueError, match=rf"{name} duration on qubits \(0,\)"):
        estimate_ibm_usage([CircuitBatch([parallel_circuit()], shots=10)], device)


def test_missing_instruction_metadata(device):
    device._backend.target.update_instruction_properties("x", (0,), None)

    with pytest.raises(ValueError, match="x duration"):
        estimate_ibm_usage([CircuitBatch([parallel_circuit()], shots=10)], device)


@pytest.mark.parametrize("value", [0, -1e-9, float("nan"), float("inf")])
def test_invalid_dt(device, value):
    device._backend.target.dt = value

    with pytest.raises(ValueError, match="dt"):
        estimate_ibm_usage([CircuitBatch([parallel_circuit()], shots=10)], device)


def test_dynamic_circuits_are_explicitly_unsupported(device):
    circuit = QuantumCircuit(3, 1)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(1)

    with pytest.raises(ValueError, match="does not support dynamic circuits"):
        estimate_ibm_usage([CircuitBatch([circuit], shots=10)], device)


def test_unknown_instruction_is_not_assumed_instantaneous(device):
    circuit = QuantumCircuit(3)
    circuit.h(0)

    with pytest.raises(ValueError, match="h duration"):
        estimate_ibm_usage([CircuitBatch([circuit], shots=10)], device)


def test_delay_in_dt_uses_backend_time_step(device):
    circuit = QuantumCircuit(3)
    circuit.delay(20_000, 0, unit="dt")

    assert estimate_ibm_usage([CircuitBatch([circuit], shots=10)], device) == pytest.approx(
        2 + 10 * (20e-6 + 7e-6 + 250e-6)
    )


def test_transpilation_failure_is_reported(device):
    device.transform.side_effect = RuntimeError("unsupported compiler input")

    with pytest.raises(ValueError, match="unsupported compiler input"):
        estimate_ibm_usage([CircuitBatch([parallel_circuit()], shots=10)], device)


@pytest.mark.parametrize("backend_name", ["FakeSherbrooke", "FakeBrisbane"])
def test_shipped_ibm_backend_metadata_supports_local_estimation(backend_name):
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import fake_provider

    backend = getattr(fake_provider, backend_name)()
    backend.run = Mock(side_effect=AssertionError("must not run a backend"))
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=0)
    device = SimpleNamespace(_backend=backend, transform=Mock(side_effect=pass_manager.run))
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    result = estimate_ibm_usage([CircuitBatch([circuit], shots=1000)], device)

    assert 2.25 < result < 2.30
    device.transform.assert_called_once()
    backend.run.assert_not_called()
