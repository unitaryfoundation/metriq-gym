"""Estimate IBM QPU usage locally from backend instruction timing metadata."""

from math import isfinite
from typing import TYPE_CHECKING, Iterable
import warnings

from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.transpiler import PassManagerConfig, Target
from qiskit.transpiler.preset_passmanagers import level_0_pass_manager

if TYPE_CHECKING:
    from qbraid import QuantumDevice

    from metriq_gym.resource_estimation import CircuitBatch


def _timing(value: object, description: str) -> float:
    """Require actual, finite timing data rather than substituting an estimate."""
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"IBM timing metadata is missing or invalid: {description}.") from None
    if not isfinite(seconds) or seconds < 0:
        raise ValueError(f"IBM timing metadata is missing or invalid: {description}.")
    return seconds


def _instruction_duration(target: Target, name: str, qubits: tuple[int, ...]) -> float:
    try:
        properties = target[name][qubits]
        duration = properties.duration
    except (KeyError, AttributeError, TypeError):
        duration = None
    return _timing(duration, f"{name} duration on qubits {qubits}")


def _validate_instructions(circuit: QuantumCircuit, target: Target) -> set[int]:
    """Check durations before scheduling, which may otherwise accept invalid data."""
    used_qubits: set[int] = set()
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, ControlFlowOp) or getattr(operation, "condition", None):
            raise ValueError("IBM static QPU time estimation does not support dynamic circuits.")
        if operation.name == "barrier":
            continue
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        used_qubits.update(qubits)
        if operation.name == "delay":
            _timing(operation.duration, "circuit delay")
            if operation.unit == "dt" and target.dt is None:
                raise ValueError("IBM timing metadata is missing: dt for a circuit delay.")
        else:
            _instruction_duration(target, operation.name, qubits)
    return used_qubits


def _scheduled_duration(circuit: QuantumCircuit, target: Target) -> float:
    """Schedule without changing the physical mapping chosen by device.transform."""
    # ALAP follows IBM's local usage example linked in estimate_ibm_usage.
    # Give Qiskit's generated stage explicit timing constraints so it includes
    # hardware alignment. Run only scheduling to preserve the physical circuit.
    config = PassManagerConfig(
        target=target,
        instruction_durations=target.durations(),
        timing_constraints=target.timing_constraints(),
        scheduling_method="alap",
    )
    scheduler = level_0_pass_manager(config).scheduling
    scheduled = scheduler.run(circuit)
    estimate_duration = getattr(scheduled, "estimate_duration", None)
    if callable(estimate_duration):
        return _timing(estimate_duration(target), "scheduled circuit duration")

    # Qiskit 1.4 does not have estimate_duration. Scheduling still populates these
    # attributes, deprecated since 1.3; access them only for the older API.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        duration = _timing(scheduled.duration, "scheduled circuit duration")
        unit = scheduled.unit
    if unit == "dt":
        return duration * _timing(target.dt, "dt")
    scales = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
    if unit not in scales:
        raise ValueError(f"IBM scheduled circuit has unsupported duration unit: {unit}.")
    return duration * scales[unit]


def estimate_ibm_usage(batches: Iterable["CircuitBatch"], device: "QuantumDevice | None") -> float:
    """Return approximate QPU seconds without submitting circuits or creating jobs.

    Follow IBM's local estimate: two seconds per execution sub-job plus shots
    times (scheduled circuit duration + initialization + repetition delay).
    Each nonempty CircuitBatch is assumed to fit in one execution sub-job. Gate
    scheduling accounts for parallel instructions, and initialization uses the
    slowest reset among the physical qubits used by that circuit, assuming resets
    can run in parallel. Default initialization and repetition delay are assumed.
    Session idle time and additional Runtime operations are not included.

    https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time
    """
    if device is None:
        raise ValueError("IBM QPU time estimation requires --device and its timing metadata.")
    backend = getattr(device, "_backend", None)
    target = getattr(backend, "target", None)
    if target is None:
        raise ValueError("IBM device does not expose backend target timing metadata.")
    rep_delay = _timing(getattr(backend, "default_rep_delay", None), "default_rep_delay")
    if target.dt is not None and _timing(target.dt, "dt") == 0:
        raise ValueError("IBM timing metadata is invalid: dt must be positive.")

    # Retain circuit references so a generator cannot recycle IDs between batches.
    durations: dict[tuple[int, bool], tuple[QuantumCircuit, float]] = {}
    total = 0.0
    for batch in batches:
        if not batch.circuits:
            continue
        if batch.shots < 0:
            raise ValueError("IBM QPU time estimation requires nonnegative shots.")
        total += 2.0
        for circuit in batch.circuits:
            cache_key = (id(circuit), batch.transpile)
            if cache_key not in durations:
                try:
                    physical = (
                        device.transform(circuit.copy()) if batch.transpile else circuit.copy()
                    )
                    if not isinstance(physical, QuantumCircuit):
                        raise ValueError("IBM local transpilation did not return a Qiskit circuit.")
                    used_qubits = _validate_instructions(physical, target)
                    init_duration = max(
                        (_instruction_duration(target, "reset", (qubit,)) for qubit in used_qubits),
                        default=0.0,
                    )
                    per_shot = _scheduled_duration(physical, target) + init_duration + rep_delay
                except Exception as exc:
                    raise ValueError(
                        f"IBM static QPU time estimation is unavailable: {exc}"
                    ) from exc
                durations[cache_key] = (circuit, per_shot)
            total += batch.shots * durations[cache_key][1]
    return total
