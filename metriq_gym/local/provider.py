from qbraid.runtime import QuantumProvider
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel

from metriq_gym.local.device import LocalAerDevice


class LocalProvider(QuantumProvider):
    def __init__(self):
        super().__init__()
        self.device = LocalAerDevice(provider=self)
        self._devices: dict[str, LocalAerDevice] = {}

    def get_devices(self, **_):
        return [self.device, *self._devices.values()]

    def get_device(self, device_id):
        if device_id in self._devices:
            return self._devices[device_id]

        if device_id == "aer_simulator":
            return self.device
        try:
            service = QiskitRuntimeService()
            backend = service.backend(device_id)
            noise_model = NoiseModel.from_backend(backend)
            aer_backend = AerSimulator.from_backend(
                backend,
                noise_model=noise_model,
                basis_gates=AerSimulator().configuration().basis_gates,
            )
        except Exception as exc:  # pragma: no cover - network exceptions
            raise ValueError("Unknown device identifier") from exc

        device = LocalAerDevice(provider=self, device_id=device_id, backend=aer_backend)
        self._devices[device_id] = device
        return device
