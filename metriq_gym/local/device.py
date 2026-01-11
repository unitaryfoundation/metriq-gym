import time
import uuid
from qbraid import QPROGRAM, load_program
from qbraid.runtime import QuantumDevice, DeviceStatus, TargetProfile
from qbraid.programs import ExperimentType, ProgramSpec
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from .job import LocalAerJob


def _make_profile(
    *, device_id: str = "aer_simulator", backend: AerSimulator | None = None
) -> TargetProfile:
    backend = backend or AerSimulator()
    cfg = backend.configuration()
    return TargetProfile(
        device_id=device_id,
        simulator=True,
        experiment_type=ExperimentType.GATE_MODEL,
        num_qubits=cfg.num_qubits,
        program_spec=ProgramSpec(QuantumCircuit),
        basis_gates=cfg.basis_gates,
        provider_name="local",
        extra={"backend": backend},
    )


class LocalAerDevice(QuantumDevice):
    def __init__(
        self, *, provider, device_id: str = "aer_simulator", backend: AerSimulator | None = None
    ) -> None:
        backend = backend or AerSimulator()
        super().__init__(_make_profile(device_id=device_id, backend=backend))
        self._backend = self.profile.extra["backend"]
        self._provider = provider

    def status(self) -> DeviceStatus:
        return DeviceStatus.ONLINE

    def transform(self, run_input):
        program = load_program(run_input)
        program.transform(self)
        return program.program

    def submit(
        self, run_input: QPROGRAM | list[QPROGRAM], *, shots: int | None = None, **kwargs
    ) -> LocalAerJob:
        start_time = time.perf_counter()
        counts = self._backend.run(run_input, shots=shots, **kwargs).result().get_counts()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return LocalAerJob(
            uuid.uuid4().hex, device=self, counts=counts, execution_time=execution_time
        )
