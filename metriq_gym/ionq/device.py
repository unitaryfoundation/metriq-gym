"""IonQ device workaround for qiskit_ionq qubit count bug.

When qiskit_ionq is installed, qBraid's IonQDevice.run() uses qiskit_ionq's
transpilation path which sets the circuit qubit count to the device's total
qubit count (e.g. 29) rather than the circuit's actual qubit count. This
causes IonQ's simulator to reject circuits that would otherwise run fine.

This module patches IonQDevice.run() to convert Qiskit QuantumCircuit inputs
to QASM2 strings before submission, using qBraid's standard conversion path
which correctly preserves the circuit's qubit count.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from qbraid.runtime import IonQDevice, IonQJob

logger = logging.getLogger(__name__)

_QISKIT_IONQ_INSTALLED = importlib.util.find_spec("qiskit_ionq") is not None


def _convert_qiskit_to_qasm2(run_input):
    """Convert Qiskit QuantumCircuit(s) to QASM2 strings if qiskit_ionq is installed.

    This bypasses qiskit_ionq's transpilation which incorrectly sets the qubit
    count to the device total rather than the circuit size.
    """
    if not _QISKIT_IONQ_INSTALLED:
        return run_input

    try:
        from qiskit import QuantumCircuit, qasm2
    except ImportError:
        return run_input

    if isinstance(run_input, list):
        if all(isinstance(p, QuantumCircuit) for p in run_input):
            return [qasm2.dumps(p) for p in run_input]
    elif isinstance(run_input, QuantumCircuit):
        return qasm2.dumps(run_input)

    return run_input


def patch_ionq_device(device: "IonQDevice") -> None:
    """Patch an IonQDevice to bypass the buggy qiskit_ionq conversion path."""
    if not _QISKIT_IONQ_INSTALLED:
        return

    original_run = device.run.__func__ if hasattr(device.run, "__func__") else None
    if original_run is None:
        return

    def patched_run(
        self,
        run_input,
        *args,
        **kwargs,
    ) -> Union["IonQJob", list["IonQJob"]]:
        # Remove qiskit_ionq-specific kwargs that don't apply to QASM path
        kwargs.pop("gateset", None)
        kwargs.pop("ionq_compiler_synthesis", None)
        run_input = _convert_qiskit_to_qasm2(run_input)
        return original_run(self, run_input, *args, **kwargs)

    import types

    device.run = types.MethodType(patched_run, device)
    logger.info("Patched IonQDevice.run() to bypass qiskit_ionq conversion")
