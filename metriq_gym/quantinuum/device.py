from __future__ import annotations

import uuid
from typing import Any, cast

from qbraid import QPROGRAM, load_program
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import DeviceStatus, QuantumDevice, TargetProfile
from qiskit import QuantumCircuit

from .job import QuantinuumJob
from .auth import load_api
from metriq_gym.local._store import write


SUPPORTED_EMULATORS = {
    # Quantinuum NEXUS targets (subset we expose by default)
    "H1-1E": {"num_qubits": 20},
    "H1-2E": {"num_qubits": 20},
    # Syntax checkers (no execution, accessibility varies)
    "H1-1SC": {"num_qubits": 20},
    "H1-2SC": {"num_qubits": 20},
}


def _make_profile(device_id: str) -> TargetProfile:
    meta = SUPPORTED_EMULATORS.get(device_id)
    if meta is None:
        raise ValueError("Unknown Quantinuum device identifier")

    return TargetProfile(
        device_id=device_id,
        simulator=True,
        experiment_type=ExperimentType.GATE_MODEL,
        num_qubits=meta["num_qubits"],
        program_spec=ProgramSpec(QuantumCircuit),
        basis_gates=None,  # Unknown; supplied by remote at runtime
        provider_name="quantinuum",
        extra={},
    )


class QuantinuumDevice(QuantumDevice):
    def __init__(self, *, provider: Any, device_id: str) -> None:
        super().__init__(_make_profile(device_id))
        self._provider = provider

    def status(self) -> DeviceStatus:
        # We expose emulators, which are typically available.
        # For production, this could query NEXUS for availability.
        return DeviceStatus.ONLINE

    def transform(self, run_input):
        program = load_program(run_input)
        program.transform(self)
        return program.program

    def run(
        self,
        run_input: QPROGRAM | list[QPROGRAM],
        *,
        shots: int | None = None,
        **kwargs: Any,
    ) -> QuantinuumJob:
        # Bypass qBraid's default transform path (which expects a Qiskit backend)
        # and directly submit to Quantinuum via pytket-quantinuum when available.

        try:
            from pytket.extensions.qiskit import qiskit_to_tk  # type: ignore
            from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency not installed
            raise RuntimeError(
                "Missing dependency: pytket-quantinuum (and pytket-qiskit). "
                "Install with: poetry add pytket-quantinuum pytket-qiskit."
            ) from exc

        # Initialize API handler via shared helper
        api: Any = load_api()

        # Normalize input to list of Qiskit circuits
        if not isinstance(run_input, list):
            run_input_list = [cast(Any, run_input)]
        else:
            run_input_list = run_input

        # Convert Qiskit circuits to pytket Circuits
        try:
            tk_circuits = [qiskit_to_tk(circ) for circ in run_input_list]
        except Exception as exc:
            raise RuntimeError("Failed to convert Qiskit circuits to pytket format.") from exc

        # Create backend for the selected device (emulator names like H1-1E / H1-2E)
        try:
            backend = QuantinuumBackend(self.id, api_handler=api)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Quantinuum backend for device '{self.id}'."
            ) from exc

        # Compile circuits for the target, then submit
        try:
            compiled = backend.get_compiled_circuits(tk_circuits)
            handles = backend.process_circuits(
                compiled,
                n_shots=shots or kwargs.get("n_shots", 100),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to submit job to Quantinuum backend: {exc}") from exc

        # Normalize handle and derive a job id string; do not fetch results here.
        handle = handles[0] if isinstance(handles, list) and handles else handles
        job_id: str = (
            getattr(handle, "job_id", None)
            or getattr(handle, "id", None)
            or str(handle)
            or uuid.uuid4().hex
        )
        return QuantinuumJob(job_id, device=self)

    # Implement abstract method to satisfy QuantumDevice ABC
    def submit(
        self,
        run_input: QPROGRAM | list[QPROGRAM],
        *,
        shots: int | None = None,
        **kwargs: Any,
    ) -> QuantinuumJob:
        return self.run(run_input, shots=shots, **kwargs)
