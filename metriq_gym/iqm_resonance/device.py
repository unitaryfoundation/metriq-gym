from __future__ import annotations

from typing import Any
import logging

from iqm.iqm_client import Circuit, Instruction
from qbraid import QPROGRAM
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import DeviceStatus, QuantumDevice, TargetProfile
from qiskit import QuantumCircuit

from metriq_gym.iqm_resonance.job import IQMResonanceJob

logger = logging.getLogger(__name__)


SUPPORTED_INSTRUCTIONS = {"phased_rx", "prx", "cz", "measurement", "measure", "barrier"}


def _profile(*, device_id: str, num_qubits: int, basis_gates: list[str]) -> TargetProfile:
    return TargetProfile(
        device_id=device_id,
        simulator=False,
        experiment_type=ExperimentType.GATE_MODEL,
        num_qubits=num_qubits,
        program_spec=ProgramSpec(QuantumCircuit),
        basis_gates=basis_gates,
        provider_name="iqm",
        extra={},
    )


class IQMResonanceDevice(QuantumDevice):
    """qBraid device wrapper for IQM hardware accessible via the iqm-client SDK."""

    def __init__(self, *, provider, device_id: str, client, architecture) -> None:
        self._provider = provider
        self._client = client
        self._arch = architecture
        super().__init__(
            _profile(
                device_id=device_id,
                num_qubits=len(getattr(architecture, "qubits", []) or architecture.qubits),
                basis_gates=list(
                    getattr(architecture, "operations", []) or architecture.operations
                ),
            )
        )

    def status(self) -> DeviceStatus:
        return DeviceStatus.ONLINE

    def _qubit_label(self, qiskit_qubit) -> str:
        idx = qiskit_qubit.index
        try:
            return self._arch.qubits[idx]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Qubit index {idx} not available on IQM device") from exc

    def _to_iqm_circuit(self, circuit: QuantumCircuit) -> tuple[Circuit, list[str]]:
        instructions: list[Instruction] = []
        measurement_keys: list[str] = []

        for inst in circuit.data:
            op = inst.operation
            name = op.name.lower()
            qargs = inst.qubits
            cargs = inst.clbits

            if name not in SUPPORTED_INSTRUCTIONS:
                raise TypeError(
                    f"Gate '{name}' not supported by IQM client integration. "
                    "Please transpile to IQM native gates (phased_rx, prx, cz, measurement)."
                )

            qubits = [self._qubit_label(q) for q in qargs]
            args: dict[str, Any] = {}

            if name in {"phased_rx", "prx"}:
                angle = float(op.params[0]) if op.params else 0.0
                phase = float(op.params[1]) if len(op.params) > 1 else 0.0
                args = {"angle_t": angle, "phase_t": phase}
            elif name in {"measurement", "measure"}:
                # Use classical bit index as measurement key for deterministic bitstrings
                if not cargs:
                    raise TypeError("Measurement requires a classical bit target on IQM devices.")
                key = f"c{cargs[0].index}"
                args = {"key": key}
                measurement_keys.append(key)
                name = "measurement"

            instructions.append(Instruction(name=name, qubits=qubits, args=args))

        iqm_circuit = Circuit(name=circuit.name or "circuit", instructions=tuple(instructions))
        return iqm_circuit, measurement_keys

    def transform(self, run_input: QPROGRAM) -> tuple[list[Circuit], list[list[str]]]:
        circuits: list[Circuit] = []
        meas_keys: list[list[str]] = []

        if isinstance(run_input, list) or isinstance(run_input, tuple):
            for item in run_input:
                c, k = self.transform(item)
                circuits.extend(c)
                meas_keys.extend(k)
            return circuits, meas_keys

        if not isinstance(run_input, QuantumCircuit):
            raise TypeError(
                f"Unsupported run_input type {type(run_input)}; expected qiskit.QuantumCircuit"
            )

        circuit: Circuit
        keys: list[str]
        circuit, keys = self._to_iqm_circuit(run_input)
        circuits.append(circuit)
        meas_keys.append(keys or [])
        return circuits, meas_keys

    def submit(
        self, run_input: QPROGRAM | list[QPROGRAM], *, shots: int | None = None, **_: Any
    ) -> IQMResonanceJob:
        circuits, meas_keys = self.transform(run_input)
        nshots = int(shots or 1000)
        job_id = self._client.submit_circuits(circuits, shots=nshots)
        return IQMResonanceJob(
            str(job_id), device=self, client=self._client, measurement_keys=meas_keys
        )
