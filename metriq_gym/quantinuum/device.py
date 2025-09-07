from __future__ import annotations

import os
import uuid
from typing import Any, cast

from qbraid import QPROGRAM, load_program
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import DeviceStatus, QuantumDevice, TargetProfile
from qiskit import QuantumCircuit

from .job import QuantinuumJob
from metriq_gym.local._store import write


SUPPORTED_EMULATORS = {
    # Quantinuum emulator device identifiers (NEXUS)
    "H1-1E": {"num_qubits": 20},
    "H1-2E": {"num_qubits": 20},
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

        # Gather credentials from environment
        raw_api_key = os.getenv("QUANTINUUM_API_KEY")
        username = os.getenv("QUANTINUUM_USERNAME")
        password = os.getenv("QUANTINUUM_PASSWORD")
        api_key = (
            raw_api_key
            if raw_api_key and not raw_api_key.strip().startswith("<") and ">" not in raw_api_key
            else None
        )

        try:
            from pytket.extensions.qiskit import qiskit_to_tk  # type: ignore
            from pytket.extensions.quantinuum import (  # type: ignore
                QuantinuumBackend,
                QuantinuumAPI,
            )
        except Exception as exc:  # pragma: no cover - optional dependency not installed
            raise RuntimeError(
                "Missing dependency: pytket-quantinuum (and pytket-qiskit). "
                "Install with: poetry add pytket-quantinuum pytket-qiskit."
            ) from exc

        # Initialize API handler (prefer username/password for broad compatibility)
        api: Any
        if username and password:
            api = QuantinuumAPI()
            # Some versions require credentials via env and a zero-arg login()
            os.environ["PYTKET_QUANTINUUM_USERNAME"] = username
            os.environ["PYTKET_QUANTINUUM_PASSWORD"] = password
            # Legacy/alternate variable names used by some releases
            os.environ["HQS_EMAIL"] = username
            os.environ["HQS_PASSWORD"] = password
            os.environ["QUANTINUUM_EMAIL"] = username
            os.environ["QUANTINUUM_PASSWORD"] = password
            if hasattr(api, "login"):
                try:
                    api.login()  # type: ignore[attr-defined]
                except TypeError:
                    # Older/newer variant expects explicit credentials
                    if hasattr(api, "set_user_credentials"):
                        api.set_user_credentials(username, password)  # type: ignore[attr-defined]
                    else:
                        raise RuntimeError(
                            "Unable to authenticate with username/password. Please update pytket-quantinuum."
                        )
                except Exception:
                    # If login still prompted/failed, proceed; backend may re-prompt interactively.
                    pass
            elif hasattr(api, "set_user_credentials"):
                api.set_user_credentials(username, password)  # type: ignore[attr-defined]
            else:
                raise RuntimeError(
                    "Unable to authenticate with username/password. Please update pytket-quantinuum."
                )
        elif api_key:
            # API key flows vary by version; guide user if unsupported in their install
            try:
                api = QuantinuumAPI(api_key=api_key)  # type: ignore[arg-type]
            except TypeError as exc:
                raise RuntimeError(
                    "Your pytket-quantinuum version does not support api_key in constructor. "
                    "Either upgrade pytket-quantinuum or use QUANTINUUM_USERNAME/QUANTINUUM_PASSWORD."
                ) from exc
        else:
            raise RuntimeError(
                "Quantinuum credentials not found. Set QUANTINUUM_USERNAME and QUANTINUUM_PASSWORD "
                "(recommended), or QUANTINUUM_API_KEY if your pytket-quantinuum supports it."
            )

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

        # Normalize handle and fetch result synchronously (emulator expected to be quick)
        handle = handles[0] if isinstance(handles, list) and handles else handles
        try:
            result = backend.get_result(handle)
            # Extract counts (single-circuit submission)
            counts = None
            get_counts = getattr(result, "get_counts", None)
            if callable(get_counts):
                try:
                    counts = get_counts(0)
                except TypeError:
                    counts = get_counts()
            if counts is None:
                counts = getattr(result, "counts", None)
            if counts is None:
                raise RuntimeError("Unable to extract measurement counts from result")
        except Exception as exc:
            raise RuntimeError("Failed to retrieve results from Quantinuum backend.") from exc

        # Persist results locally for polling via metriq-gym
        job_id = uuid.uuid4().hex
        write(
            job_id,
            {
                "job_id": job_id,
                "device_id": self.id,
                "counts": counts,
            },
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
