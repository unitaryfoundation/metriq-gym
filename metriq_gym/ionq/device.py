"""Patches for qBraid's IonQ provider.

Addresses two qBraid/IonQ issues so that every benchmark receives correctly
formatted measurement counts without benchmark-specific workarounds:

1. **Bitstring padding** — IonQJob._get_counts() uses normalize_data() which
   can return truncated bitstrings (e.g. "0" instead of "0000000") when only
   a single computational basis state is observed.  We pad every key to the
   circuit's qubit count, which is available in the IonQ job response.

2. **All-qubit measurement** — IonQ measures every qubit regardless of the
   circuit's measurement gates (qBraid's QASM→IonQ conversion discards them).
   We extract the measurement mapping at dispatch time, persist it in IonQ's
   job metadata field, and marginalize the all-qubit counts back to the
   classical register at poll time.

Note: The qiskit_ionq qubit-count bug (see qBraid/qBraid#1141) is resolved
by uninstalling qiskit-ionq, which forces qBraid to use its native
qiskit → qasm3 → ionq conversion path.
"""

import json
import logging
import types
from typing import Any

from qiskit import QuantumCircuit
from qbraid.runtime import IonQDevice
from qbraid.runtime.ionq.job import IonQJob, IonQJobError
from qbraid.runtime.postprocess import distribute_counts, normalize_data
from qbraid.runtime.result import Result
from qbraid.runtime.result_data import GateModelResultData, MeasCount

logger = logging.getLogger(__name__)

# Sentinel so we only patch the IonQJob class once per process.
_ionq_job_patched = False

# ── Metadata keys stored in IonQ job metadata ──────────────────────────
_META_MEAS_MAP = "_metriq_meas_map"
_META_NUM_CLBITS = "_metriq_num_clbits"


# ── Helpers ────────────────────────────────────────────────────────────


def _extract_measurement_map(circuit: QuantumCircuit) -> dict[str, int]:
    """Return ``{qubit_index: clbit_index}`` for every measure gate.

    Keys are strings because IonQ job metadata is JSON-serialised (int keys
    become strings).  We use strings from the start for consistency.
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
    """Reduce all-qubit counts to the classical register.

    Args:
        counts: Measurement counts with full-width (num_qubits) bitstring keys.
        meas_map: ``{"qubit_index": clbit_index}`` extracted at dispatch time.
        num_clbits: Size of the classical register.
    """
    qubit_indices = sorted(int(q) for q in meas_map)
    clbit_indices = [meas_map[str(q)] for q in qubit_indices]

    marginal: dict[str, int] = {}
    num_qubits = len(next(iter(counts)))
    for bitstring, count in counts.items():
        clbits = ["0"] * num_clbits
        for q_idx, c_idx in zip(qubit_indices, clbit_indices):
            # IonQ/qBraid bitstrings are big-endian (MSB first via bin()),
            # so qubit q is at string index (n - 1 - q).
            clbits[c_idx] = bitstring[num_qubits - 1 - q_idx]
        key = "".join(clbits)
        marginal[key] = marginal.get(key, 0) + count
    return marginal


def _apply_counts(
    counts: MeasCount | list[MeasCount],
    fn,
    *args,
) -> MeasCount | list[MeasCount]:
    """Apply *fn* to a single counts dict or to each element of a batch."""
    if isinstance(counts, list):
        return [fn(c, *args) for c in counts]
    return fn(counts, *args)


# ── Public patch entry points ──────────────────────────────────────────


def patch_ionq_device(device: IonQDevice) -> None:
    """Patch an IonQDevice instance.

    * Injects measurement-mapping metadata into the IonQ job (issue 2).
    * Ensures the class-level IonQJob patches are applied (issues 1 & 2).
    """
    patch_ionq_job()  # idempotent

    original_run = IonQDevice.run

    def run_with_patches(self, run_input, *args, **kwargs):
        # Inject measurement metadata for partial-measurement circuits.
        if isinstance(run_input, QuantumCircuit):
            if run_input.num_clbits < run_input.num_qubits:
                meas_map = _extract_measurement_map(run_input)
                metadata = kwargs.get("metadata") or {}
                metadata[_META_MEAS_MAP] = json.dumps(meas_map)
                metadata[_META_NUM_CLBITS] = str(run_input.num_clbits)
                kwargs["metadata"] = metadata

        return original_run(self, run_input, *args, **kwargs)

    device.run = types.MethodType(run_with_patches, device)


def patch_ionq_job() -> None:
    """Patch IonQJob at the class level (idempotent).

    * ``_get_counts`` — pads bitstrings to circuit qubit width (issue 2).
    * ``result`` — marginalises to the classical register when measurement
      metadata is present (issue 3).
    """
    global _ionq_job_patched  # noqa: PLW0603
    if _ionq_job_patched:
        return
    _ionq_job_patched = True

    def _get_counts_padded(result: dict[str, Any]) -> MeasCount | list[MeasCount]:
        """Replacement for IonQJob._get_counts that pads bitstrings."""
        shots = result.get("shots")
        probabilities = result.get("probabilities")
        num_qubits = result.get("qubits")

        if shots is None or probabilities is None:
            raise ValueError("Missing shots or probabilities in result data.")

        def convert_to_counts(meas_prob: dict[str, float]) -> dict[str, int]:
            probs_dec = {int(key): value for key, value in meas_prob.items()}
            probs_normal = normalize_data(probs_dec)
            counts = distribute_counts(probs_normal, shots)
            if num_qubits is not None:
                counts = _pad_counts(counts, num_qubits)
            return counts

        if all(isinstance(value, dict) for value in probabilities.values()):
            return [convert_to_counts(probs) for probs in probabilities.values()]

        return convert_to_counts(probabilities)

    def result_with_marginalization(self) -> Result:
        """Replacement for IonQJob.result that marginalises to classical bits."""
        self.wait_for_final_state()
        job_data = self.session.get_job(self.id)
        success = job_data.get("status") == "completed"
        if not success:
            failure: dict = job_data.get("failure", {})
            code = failure.get("code")
            message = failure.get("error")
            raise IonQJobError(f"Job failed with code {code}: {message}")

        results_url: str = job_data["results_url"]
        results_endpoint = results_url.split("v0.3")[-1]
        job_data["probabilities"] = self.session.get(results_endpoint).json()
        job_data["shots"] = job_data.get("shots", self._cache_metadata.get("shots"))

        measurement_counts = self._get_counts(job_data)

        # Marginalize if measurement metadata was injected at dispatch time.
        metadata = job_data.get("metadata") or {}
        meas_map_raw = metadata.get(_META_MEAS_MAP)
        num_clbits_raw = metadata.get(_META_NUM_CLBITS)
        if meas_map_raw is not None and num_clbits_raw is not None:
            meas_map = json.loads(meas_map_raw) if isinstance(meas_map_raw, str) else meas_map_raw
            num_clbits = int(num_clbits_raw)
            measurement_counts = _apply_counts(
                measurement_counts, _marginalize_to_clbits, meas_map, num_clbits
            )

        data = GateModelResultData(measurement_counts=measurement_counts)
        return Result(
            device_id=job_data["target"], job_id=self.id, success=success, data=data, **job_data
        )

    IonQJob._get_counts = staticmethod(_get_counts_padded)
    IonQJob.result = result_with_marginalization
