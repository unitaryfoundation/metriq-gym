from __future__ import annotations

import uuid
from typing import Any, cast

from qbraid import QPROGRAM, load_program
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import DeviceStatus, QuantumDevice, TargetProfile
from qiskit import QuantumCircuit

from .job import QuantinuumJob
from .auth import load_api


def _make_profile(device_id: str, num_qubits: int | None = None) -> TargetProfile:
    return TargetProfile(
        device_id=device_id,
        simulator=True,
        experiment_type=ExperimentType.GATE_MODEL,
        num_qubits=(num_qubits if isinstance(num_qubits, int) and num_qubits > 0 else 0),
        program_spec=ProgramSpec(QuantumCircuit),
        basis_gates=None,  # Provided by remote at runtime
        provider_name="quantinuum",
        extra={},
    )


class QuantinuumDevice(QuantumDevice):
    def __init__(self, *, provider: Any, device_id: str, num_qubits: int | None = None) -> None:
        super().__init__(_make_profile(device_id, num_qubits))
        self._provider = provider

    def status(self) -> DeviceStatus:
        """Return live device status from Quantinuum when possible.

        Attempts to query the Quantinuum API for the device's current state.
        Falls back to UNKNOWN if status cannot be determined.
        """
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
            api = load_api()

            # First try instance-level status accessors
            try:
                backend = QuantinuumBackend(self.id, api_handler=api)
                for attr in (
                    "device_state",
                    "device_status",
                    "get_device_status",
                    "status",
                    "get_status",
                ):
                    fn = getattr(backend, attr, None)
                    if callable(fn):
                        try:
                            val = fn()
                            if isinstance(val, str):
                                low = val.lower()
                                if any(x in low for x in ("online", "available", "ready")):
                                    return DeviceStatus.ONLINE
                                if any(x in low for x in ("offline", "unavailable")):
                                    return DeviceStatus.OFFLINE
                                if "maint" in low:
                                    # If qBraid DeviceStatus has no MAINTENANCE, treat as OFFLINE
                                    return DeviceStatus.OFFLINE
                        except TypeError:
                            # Some variants may require arguments; skip
                            pass
            except Exception:
                # If backend init fails (e.g., no access), continue to list-based discovery
                pass

            # Fallback: inspect available_devices metadata for this device
            avail = getattr(QuantinuumBackend, "available_devices", None)
            if callable(avail):
                try:
                    devices = avail(api_handler=api)
                except TypeError:
                    devices = avail()
                for item in devices or []:
                    name = None
                    status_val = None
                    if isinstance(item, str):
                        name = item
                    elif isinstance(item, dict):
                        name = item.get("name") or item.get("device_name") or item.get("label")
                        status_val = (
                            item.get("status")
                            or item.get("state")
                            or item.get("availability")
                            or item.get("is_online")
                        )
                    if name == self.id:
                        if isinstance(status_val, bool):
                            return DeviceStatus.ONLINE if status_val else DeviceStatus.OFFLINE
                        if isinstance(status_val, str):
                            low = status_val.lower()
                            if any(x in low for x in ("online", "available", "ready")):
                                return DeviceStatus.ONLINE
                            if any(x in low for x in ("offline", "unavailable")):
                                return DeviceStatus.OFFLINE
                        # If device is listed but no status flag is provided, assume unknown
                        return DeviceStatus.UNKNOWN

        except Exception:
            # On any error, prefer an honest UNKNOWN rather than assuming ONLINE
            pass
        return DeviceStatus.UNKNOWN

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
