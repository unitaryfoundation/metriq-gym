"""IonQ device workaround for qiskit_ionq qubit count bug.

When qiskit_ionq is installed, qBraid's IonQDevice.run() uses qiskit_ionq's
transpilation path which sets the circuit qubit count to the device's total
qubit count (e.g. 29) rather than the circuit's actual qubit count. This
causes IonQ's simulator to reject circuits that would otherwise run fine.

We work around this by converting Qiskit QuantumCircuit inputs to QASM2
strings before they reach IonQDevice.run(), which routes them through
qBraid's standard QASM-to-IonQ conversion instead.
"""

import types

from qiskit import QuantumCircuit, qasm2
from qbraid.runtime import IonQDevice


def _circuits_to_qasm2(run_input):
    """Convert Qiskit QuantumCircuit(s) to QASM2 strings."""
    if isinstance(run_input, QuantumCircuit):
        return qasm2.dumps(run_input)
    if isinstance(run_input, list) and all(isinstance(p, QuantumCircuit) for p in run_input):
        return [qasm2.dumps(p) for p in run_input]
    return run_input


def patch_ionq_device(device: IonQDevice) -> None:
    """Wrap IonQDevice.run() to convert Qiskit circuits to QASM2 before submission."""
    original_run = IonQDevice.run

    def run_with_qasm_conversion(self, run_input, *args, **kwargs):
        kwargs.pop("gateset", None)
        kwargs.pop("ionq_compiler_synthesis", None)
        return original_run(self, _circuits_to_qasm2(run_input), *args, **kwargs)

    device.run = types.MethodType(run_with_qasm_conversion, device)
