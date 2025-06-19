import uuid
from qbraid import QPROGRAM
from qbraid.runtime import QuantumDevice, DeviceStatus, TargetProfile
from qbraid.programs import ExperimentType, ProgramSpec
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from .job import LocalAerJob


def _make_profile() -> TargetProfile:
    device_id = "aer_simulator"
    backend = AerSimulator()
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
    def __init__(self, *, provider):
        super().__init__(_make_profile())
        self._backend = self.profile.extra["backend"]
        self._provider = provider

    def status(self):
        return DeviceStatus.ONLINE

    def submit(
        self, run_input: QPROGRAM | list[QPROGRAM], *, shots: int | None = None, **kwargs
    ) -> LocalAerJob:
        counts = self._backend.run(run_input, shots=shots, **kwargs).result().get_counts()
        return LocalAerJob(uuid.uuid4().hex, device=self, counts=counts)
