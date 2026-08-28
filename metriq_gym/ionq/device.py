"""Patches for qBraid's IonQ provider (qbraid >= 0.12, with qiskit-ionq).

Two adjustments make every benchmark receive correctly shaped measurement
counts without benchmark-specific workarounds:

1. **Native conversion path.** With qiskit-ionq installed, qBraid transpiles
   qiskit circuits with qiskit's default ``optimization_level=2``, which lays
   the logical qubits out onto arbitrary physical qubits of the (29-qubit)
   IonQ backend. Measured bitstrings then no longer line up with the circuit's
   own qubits (e.g. logical qubit 3 surfaces at physical qubit 28). We convert
   each circuit to OpenQASM 3 before dispatch, which routes through qBraid's
   native qasm->ionq path and preserves qubit indices (logical qubit i is
   bit i of the result).

2. **Count reshaping.** IonQ returns probabilities keyed by integer, which
   qBraid normalizes to bitstrings whose width is only that of the largest
   observed value (so a circuit that only ever yields |0..0> and |0..01>
   reports "0"/"1"). We pad every bitstring to the circuit's qubit count and
   marginalize onto its classical register, using measurement metadata
   persisted in the IonQ job at dispatch time.

The dispatch (patch_ionq_device) and poll (patch_ionq_job) sides communicate
through the IonQ job's ``metadata`` field, which survives across processes, so
the same reshaping applies whether polling in the dispatching process or a
later one.
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

# Keys under which per-circuit reshaping metadata is stored in the IonQ job.
# Each holds a JSON list with one entry per dispatched circuit, in order.
_META_MEAS_MAPS = "_metriq_meas_maps"
_META_NUM_QUBITS = "_metriq_num_qubits"
_META_NUM_CLBITS = "_metriq_num_clbits"


# ── Helpers ────────────────────────────────────────────────────────────


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


def _pad_counts(counts: MeasCount, num_qubits: int) -> MeasCount:
    """Zero-pad every bitstring key to *num_qubits* characters."""
    return {k.zfill(num_qubits): v for k, v in counts.items()}


def _marginalize_to_clbits(
    counts: MeasCount,
    meas_map: dict[str, int],
    num_clbits: int,
) -> MeasCount:
    """Reduce full-width qubit counts to the circuit's classical register.

    Args:
        counts: Measurement counts with full-width (num_qubits) bitstring keys.
        meas_map: ``{"qubit_index": clbit_index}`` extracted at dispatch time.
        num_clbits: Size of the classical register.
    """
    qubit_indices = sorted(int(q) for q in meas_map)
    clbit_indices = [meas_map[str(q)] for q in qubit_indices]

    marginal: dict[str, int] = {}
    for bitstring, count in counts.items():
        width = len(bitstring)
        # Bitstrings are big-endian: index i holds bit/clbit (n - 1 - i), so an
        # identity qubit->clbit map reproduces the input unchanged.
        clbits = ["0"] * num_clbits
        for q_idx, c_idx in zip(qubit_indices, clbit_indices):
            clbits[num_clbits - 1 - c_idx] = bitstring[width - 1 - q_idx]
        key = "".join(clbits)
        marginal[key] = marginal.get(key, 0) + count
    return marginal


def _reshape_counts(
    counts: MeasCount | list[MeasCount],
    meas_maps: list[dict[str, int]] | None,
    num_qubits: list[int] | None,
    num_clbits: list[int] | None,
) -> MeasCount | list[MeasCount]:
    """Pad, then marginalize, each circuit's counts using dispatch metadata.

    ``counts`` is a single dict for a single-circuit job or a list (in circuit
    dispatch order) for a multi-circuit job. When metadata is absent the counts
    are returned unchanged.
    """
    if num_qubits is None:
        return counts

    as_list = isinstance(counts, list)
    counts_list: list[MeasCount] = counts if as_list else [counts]  # type: ignore[assignment]

    reshaped: list[MeasCount] = []
    for i, circuit_counts in enumerate(counts_list):
        nq = num_qubits[i] if i < len(num_qubits) else num_qubits[-1]
        padded = _pad_counts(circuit_counts, nq)
        if meas_maps is not None and num_clbits is not None:
            padded = _marginalize_to_clbits(padded, meas_maps[i], num_clbits[i])
        reshaped.append(padded)

    return reshaped if as_list else reshaped[0]


# ── Public patch entry points ──────────────────────────────────────────


def patch_ionq_device(device: IonQDevice) -> None:
    """Patch an IonQDevice instance.

    Converts qiskit circuits to OpenQASM 3 (forcing qBraid's native path) and
    records per-circuit reshaping metadata on the IonQ job. Also ensures the
    class-level IonQJob patch is applied so polling reshapes the counts.
    """
    patch_ionq_job()  # idempotent

    original_run = device.run

    def run_with_patches(run_input, *args, **kwargs):
        is_single = not isinstance(run_input, list)
        circuits = [run_input] if is_single else list(run_input)

        # Only qiskit circuits need the native-path conversion and reshaping.
        if circuits and all(isinstance(c, QuantumCircuit) for c in circuits):
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
    pads and marginalizes the counts using the metadata recorded at dispatch.
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
        if _META_NUM_QUBITS not in metadata:
            return result  # not dispatched through metriq-gym; leave as-is

        meas_maps = json.loads(metadata[_META_MEAS_MAPS]) if _META_MEAS_MAPS in metadata else None
        num_qubits = json.loads(metadata[_META_NUM_QUBITS])
        num_clbits = json.loads(metadata[_META_NUM_CLBITS]) if _META_NUM_CLBITS in metadata else None

        reshaped = _reshape_counts(
            result.data.get_counts(), meas_maps, num_qubits, num_clbits
        )
        data = GateModelResultData(measurement_counts=reshaped)
        return Result(
            device_id=result.device_id,
            job_id=result.job_id,
            success=result.success,
            data=data,
        )

    IonQJob.result = result_reshaped
