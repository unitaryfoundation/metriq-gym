import uuid
from qbraid import QPROGRAM
from qbraid.runtime import QuantumDevice, DeviceStatus, TargetProfile
from qbraid.programs import ExperimentType, ProgramSpec
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from .job import LocalAerJob


def _make_profile(
    backend: AerSimulator | None = None, device_id: str = "aer_simulator"
) -> TargetProfile:
    """Creates a TargetProfile for a given backend or a default one."""
    # If no backend is provided, create a default AerSimulator.
    if backend is None:
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
    def __init__(
        self,
        *,
        provider,
        backend: AerSimulator | None = None,
        device_id: str = "aer_simulator",
    ):
        """Initializes the device, accepting an optional backend."""
        # Create the profile using our new flexible function
        profile = _make_profile(backend=backend, device_id=device_id)
        super().__init__(profile)
        self._backend = self.profile.extra["backend"]
        self._provider = provider

    def status(self):
        return DeviceStatus.ONLINE

    def submit(
        self, run_input: QPROGRAM | list[QPROGRAM], *, shots: int | None = None, **kwargs
    ) -> LocalAerJob:
        counts = self._backend.run(run_input, shots=shots, **kwargs).result().get_counts()
        return LocalAerJob(uuid.uuid4().hex, device=self, counts=counts)
