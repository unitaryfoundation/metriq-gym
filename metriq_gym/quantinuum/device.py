from typing import Any
import os
import uuid
from datetime import datetime

import qnexus as qnx
from qnexus.models.language import Language
from pytket import Circuit
try:
    from pytket.extensions.qiskit import qiskit_to_tk
except Exception:  # keep module import at top; handle absence gracefully in transform
    qiskit_to_tk = None  # type: ignore
from qiskit import QuantumCircuit

from qbraid import QPROGRAM
from qbraid.runtime import DeviceStatus, QuantumDevice, TargetProfile
from qbraid.programs import ExperimentType, ProgramSpec
from .job import QuantinuumJob


def _profile(device_id: str) -> TargetProfile:
    return TargetProfile(
        device_id=device_id,
        simulator="E" in device_id.upper(),
        experiment_type=ExperimentType.GATE_MODEL,
        num_qubits=0,
        program_spec=ProgramSpec(Circuit),
        basis_gates=None,
        provider_name="quantinuum",
        extra={},
    )


class QuantinuumDevice(QuantumDevice):
    def __init__(self, *, provider, device_id: str) -> None:
        super().__init__(_profile(device_id))
        self._provider = provider

    def status(self) -> DeviceStatus:
        return DeviceStatus.ONLINE

    def transform(self, run_input: Any):
        if isinstance(run_input, Circuit):
            return run_input
        if isinstance(run_input, list):
            return [self.transform(item) for item in run_input]
        if isinstance(run_input, QuantumCircuit):
            if qiskit_to_tk is None:
                raise ImportError(
                    "pytket-qiskit is required for Qiskit→pytket conversion; install a compatible version."
                )
            return qiskit_to_tk(run_input)  # type: ignore
        raise TypeError(
            f"Unsupported run_input type {type(run_input)}; expected pytket.Circuit or qiskit.QuantumCircuit"
        )

    def submit(self, run_input: QPROGRAM, *, shots: int | None = None, **_: Any) -> QuantinuumJob:
        project = qnx.projects.get_or_create(name=os.getenv("QNEXUS_PROJECT_NAME", "metriq-gym"))
        backend_config = qnx.QuantinuumConfig(device_name=self.profile.device_id)

        circuits = self.transform(run_input)
        circuits_list = circuits if isinstance(circuits, list) else [circuits]

        def unique(label: str) -> str:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            return f"metriq-gym {label} {ts}-{uuid.uuid4().hex[:6]}"

        circuit_refs = [
            qnx.circuits.upload(name=unique(f"circuit-{i}"), circuit=c, project=project)
            for i, c in enumerate(circuits_list)
        ]

        opt = int(os.getenv("QNEXUS_OPT_LEVEL", "1"))
        compile_job = qnx.start_compile_job(
            programs=circuit_refs,
            name=unique("compile"),
            optimisation_level=opt,
            backend_config=backend_config,
            project=project,
        )
        qnx.jobs.wait_for(compile_job)
        compiled_refs = [item.get_output() for item in qnx.jobs.results(compile_job)]

        nshots = int(shots or 1000)
        execute_job = qnx.start_execute_job(
            programs=compiled_refs,
            name=unique("execute"),
            n_shots=[nshots] * len(compiled_refs),
            backend_config=backend_config,
            project=project,
            language=Language.QIR,
        )
        job_id = getattr(execute_job, "id", None) or getattr(execute_job, "job_id", None) or str(execute_job)
        return QuantinuumJob(str(job_id), device=self)
