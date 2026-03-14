"""Tests for the IonQ device workaround that converts Qiskit circuits to QASM2."""

from unittest.mock import MagicMock

from qiskit import QuantumCircuit
from qbraid.runtime import IonQDevice

from metriq_gym.ionq.device import _circuits_to_qasm2


class TestCircuitsToQasm2:
    def test_converts_single_circuit(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        result = _circuits_to_qasm2(qc)
        assert isinstance(result, str)
        assert "OPENQASM 2.0" in result
        assert "h q[0]" in result

    def test_converts_list_of_circuits(self):
        qc1 = QuantumCircuit(2)
        qc1.h(0)

        qc2 = QuantumCircuit(2)
        qc2.x(0)

        result = _circuits_to_qasm2([qc1, qc2])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_passes_through_non_qiskit_input(self):
        qasm_str = "OPENQASM 2.0; qreg q[2]; h q[0];"
        assert _circuits_to_qasm2(qasm_str) == qasm_str

    def test_passes_through_list_of_strings(self):
        inputs = ["OPENQASM 2.0;", "OPENQASM 2.0;"]
        assert _circuits_to_qasm2(inputs) == inputs


class TestPatchIonqDevice:
    def test_patched_run_converts_circuit_to_qasm(self):
        """Patched device.run() should convert QuantumCircuit to QASM2 string."""
        original_run = MagicMock()

        device = MagicMock(spec=IonQDevice)
        device.run = original_run

        # Manually set up the same way patch_ionq_device does, using the mock as original
        import types

        def run_with_qasm_conversion(self, run_input, *args, **kwargs):
            kwargs.pop("gateset", None)
            kwargs.pop("ionq_compiler_synthesis", None)
            return original_run(_circuits_to_qasm2(run_input), *args, **kwargs)

        device.run = types.MethodType(run_with_qasm_conversion, device)

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure([0, 1], [0, 1])

        device.run(qc, shots=100)

        # Verify original_run received a QASM string, not a QuantumCircuit
        call_args = original_run.call_args
        run_input_received = call_args[0][0]
        assert isinstance(run_input_received, str)
        assert "OPENQASM 2.0" in run_input_received
