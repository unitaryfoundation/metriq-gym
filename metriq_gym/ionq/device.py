"""Patches for qBraid's IonQ provider.

IonQ measures every qubit of a submitted circuit and qBraid's conversion to
IonQ's JSON drops the measurement gates, so a circuit that measures a subset
of its qubits comes back with counts over all of them. A 7-qubit WIT circuit
that measures one qubit returns 7-bit keys where the benchmark expects 1-bit
keys. qBraid has no record of which qubits the circuit measured, so the
marginalization has to happen here: ``patch_ionq_device`` stores each
circuit's qubit-to-clbit map in the IonQ job's ``metadata`` at dispatch, and
``patch_ionq_job`` reads it back when the result is fetched. The metadata is
kept by IonQ, so polling from a later process sees the same map.

Circuits also go out as OpenQASM 3 rather than qiskit objects. With
qiskit-ionq installed, qBraid otherwise runs ``qiskit.transpile`` at the
default optimization level 2, which qiskit-ionq itself warns re-synthesizes
circuits aggressively. Benchmarks should reach the device as written.

qBraid 0.13 pads IonQ counts to the submitted register width and maps
transpiled circuits back onto their original qubit indices, so neither is
handled here any more.
"""

import json
import logging
from typing import Any

from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as qasm3_dumps
from qbraid.runtime import IonQDevice
from qbraid.runtime.ionq.job import IonQJob
from qbraid.runtime.result import Result
from qbraid.runtime.result_data import GateModelResultData, MeasCount

logger = logging.getLogger(__name__)

# Sentinel so we only patch the IonQJob class once per process.
_ionq_job_patched = False

# Keys under which per-circuit measurement metadata is stored in the IonQ job.
# Each holds a JSON list with one entry per dispatched circuit, in order.
_META_MEAS_MAPS = "_metriq_meas_maps"
_META_NUM_QUBITS = "_metriq_num_qubits"
_META_NUM_CLBITS = "_metriq_num_clbits"


# Non-unitary operations that qBraid's OpenQASM-to-IonQ conversion drops without
# warning. IonQ's API has no mid-circuit reset or measurement, so a circuit that
# uses one cannot run as written and must not report a result as if it had.
_UNSUPPORTED_OPERATIONS = frozenset({"reset"})

# Operations that touch a qubit without changing its state, so they are fine
# after a measurement.
_INERT_OPERATIONS = frozenset({"barrier", "delay"})


def _reject_unsupported_operations(circuits: list[QuantumCircuit]) -> None:
    """Raise if any circuit uses an operation IonQ cannot run.

    IonQ measures every qubit once, at the end. A measurement is therefore only
    faithful when nothing acts on that qubit afterwards; otherwise the poll
    patch reads the qubit's final value, not the one the circuit recorded.
    """
    for index, circuit in enumerate(circuits):
        label = f"Circuit {index} ({circuit.name!r})"
        measured: set[int] = set()
        for inst in circuit.data:
            name = inst.operation.name
            if name in _UNSUPPORTED_OPERATIONS:
                raise ValueError(
                    f"{label} uses '{name}', which IonQ does not support and qBraid's "
                    "conversion would silently drop."
                )
            if name in _INERT_OPERATIONS:
                continue
            qubits = [circuit.find_bit(q).index for q in inst.qubits]
            touched = measured.intersection(qubits)
            if touched:
                raise ValueError(
                    f"{label} applies '{name}' to qubit {min(touched)} after measuring it. "
                    "IonQ measures only at the end of the circuit, so mid-circuit "
                    "measurement cannot run as written."
                )
            if name == "measure":
                measured.update(qubits)


def _extract_measurement_map(circuit: QuantumCircuit) -> dict[str, int]:
    """Return ``{"qubit_index": clbit_index}`` for every measure gate.

    Keys are strings because the map is JSON-serialised into IonQ job
    metadata (JSON object keys are always strings).
    """
    meas_map: dict[str, int] = {}
    for inst in circuit.data:
        if inst.operation.name == "measure":
            q_idx = circuit.find_bit(inst.qubits[0]).index
            c_idx = circuit.find_bit(inst.clbits[0]).index
            meas_map[str(q_idx)] = c_idx
    return meas_map


def _marginalize_to_clbits(
    counts: MeasCount,
    meas_map: dict[str, int],
    num_qubits: int,
    num_clbits: int,
) -> MeasCount:
    """Reduce counts over every qubit to the circuit's classical register.

    Args:
        counts: Counts keyed by bitstring over all ``num_qubits`` qubits.
        meas_map: ``{"qubit_index": clbit_index}`` extracted at dispatch time.
        num_qubits: Width of the circuit's own register, used as a floor on the
            key width for the case where IonQ omits the batch width and qBraid
            falls back to the widest observed outcome.
        num_clbits: Size of the classical register.
    """
    qubit_indices = sorted(int(q) for q in meas_map)
    clbit_indices = [meas_map[str(q)] for q in qubit_indices]

    marginal: dict[str, int] = {}
    for bitstring, count in counts.items():
        # IonQ runs a multi-circuit job at the width of its widest circuit and
        # qBraid pads every histogram to that shared width, so a key can be
        # wider than this circuit's own register. Qubit 0 is always the last
        # character, so index from the right end of whatever width arrived.
        bits = bitstring.zfill(num_qubits)
        width = len(bits)
        clbits = ["0"] * num_clbits
        for q_idx, c_idx in zip(qubit_indices, clbit_indices):
            clbits[num_clbits - 1 - c_idx] = bits[width - 1 - q_idx]
        key = "".join(clbits)
        marginal[key] = marginal.get(key, 0) + count
    return marginal


def _reshape_counts(
    counts: MeasCount | list[MeasCount],
    meas_maps: list[dict[str, int]],
    num_qubits: list[int],
    num_clbits: list[int],
) -> MeasCount | list[MeasCount]:
    """Marginalize each circuit's counts using the metadata recorded at dispatch.

    ``counts`` is a single dict for a single-circuit job or a list (in circuit
    dispatch order) for a multi-circuit job.
    """
    as_list = isinstance(counts, list)
    counts_list: list[MeasCount] = counts if as_list else [counts]  # type: ignore[assignment]

    reshaped = [
        _marginalize_to_clbits(circuit_counts, meas_maps[i], num_qubits[i], num_clbits[i])
        for i, circuit_counts in enumerate(counts_list)
    ]
    return reshaped if as_list else reshaped[0]


def patch_ionq_device(device: IonQDevice) -> None:
    """Patch an IonQDevice instance.

    Rejects circuits IonQ cannot run, converts qiskit circuits to OpenQASM 3,
    and records per-circuit measurement metadata on the IonQ job. Also ensures the class-level IonQJob patch is
    applied so polling marginalizes the counts.
    """
    patch_ionq_job()  # idempotent

    original_run = device.run

    def run_with_patches(run_input, *args, **kwargs):
        is_single = not isinstance(run_input, list)
        circuits = [run_input] if is_single else list(run_input)

        # Only qiskit circuits carry a classical register to marginalize onto.
        if circuits and all(isinstance(c, QuantumCircuit) for c in circuits):
            _reject_unsupported_operations(circuits)
            meas_maps = [_extract_measurement_map(c) for c in circuits]
            num_qubits = [c.num_qubits for c in circuits]
            num_clbits = [c.num_clbits for c in circuits]

            metadata = dict(kwargs.get("metadata") or {})
            metadata[_META_MEAS_MAPS] = json.dumps(meas_maps)
            metadata[_META_NUM_QUBITS] = json.dumps(num_qubits)
            metadata[_META_NUM_CLBITS] = json.dumps(num_clbits)
            kwargs["metadata"] = metadata

            converted: list[Any] = [qasm3_dumps(c) for c in circuits]
            run_input = converted[0] if is_single else converted

        return original_run(run_input, *args, **kwargs)

    device.run = run_with_patches


def patch_ionq_job() -> None:
    """Patch IonQJob.result at the class level (idempotent).

    Wraps qBraid's own ``result`` so it keeps tracking qBraid internals, then
    marginalizes the counts using the metadata recorded at dispatch.
    """
    global _ionq_job_patched  # noqa: PLW0603
    if _ionq_job_patched:
        return
    _ionq_job_patched = True

    original_result = IonQJob.result

    def result_reshaped(self) -> Result:
        result = original_result(self)

        job_data = self.session.get_job(self.id) or {}
        metadata = job_data.get("metadata") or {}
        if _META_MEAS_MAPS not in metadata:
            return result  # not dispatched through metriq-gym; leave as-is

        reshaped = _reshape_counts(
            result.data.get_counts(),
            json.loads(metadata[_META_MEAS_MAPS]),
            json.loads(metadata[_META_NUM_QUBITS]),
            json.loads(metadata[_META_NUM_CLBITS]),
        )
        data = GateModelResultData(measurement_counts=reshaped)
        return Result(
            device_id=result.device_id,
            job_id=result.job_id,
            success=result.success,
            data=data,
        )

    IonQJob.result = result_reshaped
