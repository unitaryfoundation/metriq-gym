from __future__ import annotations

import os
from typing import Any
import uuid
from datetime import datetime

from qbraid import QPROGRAM
from qbraid.runtime import DeviceStatus, QuantumDevice, TargetProfile
from qbraid.programs import ExperimentType, ProgramSpec


def _make_profile(*, device_id: str) -> TargetProfile:
    try:
        from pytket import Circuit  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "pytket is required for Quantinuum provider. Install the 'quantinuum' extras: `poetry install --extras quantinuum` (use Python 3.12)."
        ) from exc

    # num_qubits, basis_gates can be discovered via qnexus, but are optional for profile creation
    return TargetProfile(
        device_id=device_id,
        simulator="E" in device_id.upper(),  # heuristic: H1-1LE/H1-Emulator
        experiment_type=ExperimentType.GATE_MODEL,
        num_qubits=0,  # unknown; not strictly required by qbraid runtime
        program_spec=ProgramSpec(Circuit),
        basis_gates=None,
        provider_name="quantinuum",
        extra={},
    )


class QuantinuumDevice(QuantumDevice):
    def __init__(self, *, provider, device_id: str) -> None:
        super().__init__(_make_profile(device_id=device_id))
        self._provider = provider

    def status(self) -> DeviceStatus:  # pragma: no cover - network status
        # Best-effort: if credentials are present, assume ONLINE; otherwise UNKNOWN
        if os.getenv("QUANTINUUM_API_KEY") or (
            os.getenv("QUANTINUUM_USERNAME") and os.getenv("QUANTINUUM_PASSWORD")
        ):
            return DeviceStatus.ONLINE
        return DeviceStatus.UNKNOWN

    def transform(self, run_input: Any):  # noqa: ANN401
        """Transform input program to a pytket Circuit accepted by Quantinuum.

        Accepts:
        - pytket `Circuit` (returned as-is)
        - qiskit `QuantumCircuit` (converted via pytket-qiskit)
        """
        try:
            from pytket import Circuit  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pytket is required for Quantinuum provider. Install with `poetry add pytket pytket-qiskit`."
            ) from exc

        if isinstance(run_input, Circuit):
            return run_input

        if isinstance(run_input, list):
            return [self.transform(item) for item in run_input]

        # Try Qiskit -> pytket
        try:
            from qiskit import QuantumCircuit  # type: ignore
            from pytket.extensions.qiskit import qiskit_to_tk  # type: ignore
        except Exception:
            QuantumCircuit = None  # type: ignore
            qiskit_to_tk = None  # type: ignore

        if QuantumCircuit is not None and isinstance(run_input, QuantumCircuit):
            if qiskit_to_tk is None:
                raise RuntimeError(
                    "pytket-qiskit is required to convert Qiskit circuits. Install the 'quantinuum' extras: `poetry install --extras quantinuum` (use Python 3.12)."
                )
            return qiskit_to_tk(run_input)

        # As a fallback, allow qbraid program loader to attempt transformation
        try:  # pragma: no cover - relies on qbraid adapters
            from qbraid import load_program

            program = load_program(run_input)
            program.transform(self)
            return program.program
        except Exception as exc:
            raise TypeError(
                f"Unsupported run_input type {type(run_input)}; expected pytket.Circuit or qiskit.QuantumCircuit"
            ) from exc

    def submit(self, run_input: QPROGRAM, *, shots: int | None = None, **kwargs) -> "QuantinuumJob":
        """Compile the circuit with TKET on NEXUS, then execute on the target device.

        Returns a `QuantinuumJob` that can be polled via the metriq-gym CLI.
        """
        from .job import QuantinuumJob

        try:
            import qnexus as qnx  # type: ignore
            from qnexus.models.language import Language  # type: ignore
            try:
                from qnexus.exceptions import AuthenticationError, ResourceCreateFailed  # type: ignore
            except Exception:  # pragma: no cover
                class AuthenticationError(Exception):
                    pass

                class ResourceCreateFailed(Exception):
                    pass
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "qnexus is required to submit jobs. Install the 'quantinuum' extras: `poetry install --extras quantinuum` (use Python 3.12)."
            ) from exc

        circuit_or_list = self.transform(run_input)

        # Authenticate using env vars if necessary
        from .utils import ensure_login
        ensure_login()

        def _auth_retry(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except AuthenticationError:
                ensure_login()
                return fn(*args, **kwargs)

        project_name = os.getenv("QNEXUS_PROJECT_NAME", "metriq-gym")
        # Ensure we can retrieve or create the project; if a conflict occurs, try to find existing by listing
        try:
            project_ref = _auth_retry(qnx.projects.get_or_create, name=project_name)
        except ResourceCreateFailed:
            # Attempt to locate an existing project with the same name
            try:
                projects_df = _auth_retry(qnx.projects.list).df()  # type: ignore[attr-defined]
                matches = projects_df[projects_df["name"] == project_name]
                if not matches.empty:
                    # Many SDKs allow get by id/reference
                    project_ref = qnx.projects.get(matches.iloc[0]["id"])  # type: ignore[index]
                else:
                    # Last resort, create a uniquely named project
                    unique_name = f"{project_name}-{uuid.uuid4().hex[:8]}"
                    project_ref = _auth_retry(qnx.projects.get_or_create, name=unique_name)
            except Exception:
                # Fallback unique create
                unique_name = f"{project_name}-{uuid.uuid4().hex[:8]}"
                project_ref = _auth_retry(qnx.projects.get_or_create, name=unique_name)

        device_name = self.profile.device_id

        # Optional: pass through error params via env to support emulator configs
        backend_config = qnx.QuantinuumConfig(device_name=device_name)

        # Helper to build a unique name to avoid 409 conflicts on the platform
        name_prefix = os.getenv("QNEXUS_NAME_PREFIX", "metriq-gym")
        def _unique(label: str) -> str:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            return f"{name_prefix} {label} {ts}-{uuid.uuid4().hex[:6]}"

        # Upload single circuit or list of circuits
        if isinstance(circuit_or_list, list):
            circuit_refs = [
                _auth_retry(
                    qnx.circuits.upload,
                    name=_unique(f"circuit-{i}"),
                    circuit=c,
                    project=project_ref,
                )
                for i, c in enumerate(circuit_or_list)
            ]
        else:
            circuit_refs = [
                _auth_retry(
                    qnx.circuits.upload,
                    name=_unique("circuit"),
                    circuit=circuit_or_list,
                    project=project_ref,
                )
            ]

        # Helper: start compile job (signature varies across qnexus versions)
        def _start_compile(program_refs):
            opt_level = int(os.getenv("QNEXUS_OPT_LEVEL", "1"))
            # Try canonical signature first
            job_name = _unique("compile")
            try:
                return qnx.start_compile_job(
                    programs=program_refs,
                    name=job_name,
                    optimisation_level=opt_level,
                    backend_config=backend_config,
                    project=project_ref,
                )
            except TypeError:
                # Try positional first-arg form
                try:
                    return qnx.start_compile_job(
                        program_refs,
                        name=job_name,
                        optimisation_level=opt_level,
                        backend_config=backend_config,
                        project=project_ref,
                    )
                except TypeError:
                    # Try US spelling
                    try:
                        return qnx.start_compile_job(
                            programs=program_refs,
                            name=job_name,
                            optimization_level=opt_level,
                            backend_config=backend_config,
                            project=project_ref,
                        )
                    except TypeError:
                        # Last fallback: alternate kwarg for list parameter
                        return qnx.start_compile_job(
                            items=program_refs,
                            name=job_name,
                            optimization_level=opt_level,
                            backend_config=backend_config,
                            project=project_ref,
                        )

        shots = int(shots or kwargs.get("n_shots", 1000))

        # Helper: start execute job (signature varies across qnexus versions)
        def _start_execute(program_refs, shots_list):
            job_name = _unique("execute")
            try:
                return qnx.start_execute_job(
                    programs=program_refs,
                    name=job_name,
                    n_shots=shots_list,
                    backend_config=backend_config,
                    project=project_ref,
                    language=Language.QIR,
                )
            except TypeError:
                try:
                    return qnx.start_execute_job(
                        program_refs,
                        name=job_name,
                        n_shots=shots_list,
                        backend_config=backend_config,
                        project=project_ref,
                        language=Language.QIR,
                    )
                except TypeError:
                    try:
                        return qnx.start_execute_job(
                            items=program_refs,
                            name=job_name,
                            n_shots=shots_list,
                            backend_config=backend_config,
                            project=project_ref,
                            language=Language.QIR,
                        )
                    except TypeError:
                        # Final fallback: drop language kwarg entirely
                        try:
                            return qnx.start_execute_job(
                                programs=program_refs,
                                name=job_name,
                                n_shots=shots_list,
                                backend_config=backend_config,
                                project=project_ref,
                            )
                        except TypeError:
                            return qnx.start_execute_job(
                                program_refs,
                                name=job_name,
                                n_shots=shots_list,
                                backend_config=backend_config,
                                project=project_ref,
                            )

        # Prefer compile+execute by default to satisfy backend gate set; optionally try direct execute first
        execute_job = None
        if os.getenv("QNEXUS_COMPILE_FIRST", "1") == "0":
            try:
                execute_job = _auth_retry(_start_execute, circuit_refs, [shots] * len(circuit_refs))
            except Exception:
                execute_job = None

        if execute_job is None:
            compile_job = _auth_retry(_start_compile, circuit_refs)
            _auth_retry(qnx.jobs.wait_for, compile_job)
            compiled_refs = [
                item.get_output() for item in _auth_retry(qnx.jobs.results, compile_job)
            ]
            execute_job = _auth_retry(
                _start_execute, compiled_refs, [shots] * len(compiled_refs)
            )

        # qnexus returns a JobRef-like object; use its ID string for persistence
        job_id = getattr(execute_job, "id", None) or getattr(execute_job, "job_id", None)
        if job_id is None:
            job_id = str(execute_job)

        return QuantinuumJob(job_id, device=self)
