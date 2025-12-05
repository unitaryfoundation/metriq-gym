from qbraid.runtime import QuantumProvider
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from metriq_gym.local.device import LocalAerDevice


class LocalProvider(QuantumProvider):
    def __init__(self):
        super().__init__()
        self.device = LocalAerDevice(provider=self)
        self._devices: dict[str, LocalAerDevice] = {}

    def get_devices(self, **_) -> list[QuantumDevice]:
        devices = [self.device]

        # Try to get IBM backends if QiskitRuntimeService is configured
        try:
            service = QiskitRuntimeService()
            backends = service.backends()
            # Add available backends to the list
            for backend in backends:
                device_id = backend.name
                if device_id != "aer_simulator":
                    try:
                        # Get or create the device (this will cache it in self._devices)
                        device = self.get_device(device_id)
                        if device not in devices:
                            devices.append(device)
                    except Exception:
                        # Skip backends that can't be loaded
                        pass
        except Exception:
            # QiskitRuntimeService not configured or other error - just continue
            pass

        return devices

    def get_device(self, device_id) -> QuantumDevice:
        if device_id in self._devices:
            return self._devices[device_id]

        if device_id == "aer_simulator":
            return self.device
        try:
            aer_backend = AerSimulator.from_backend(QiskitRuntimeService().backend(device_id))
        except Exception as exc:  # pragma: no cover - network exceptions
            raise ValueError("Unknown device identifier") from exc

        device = LocalAerDevice(provider=self, device_id=device_id, backend=aer_backend)
        self._devices[device_id] = device
        return device
