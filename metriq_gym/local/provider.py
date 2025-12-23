from qbraid.runtime import QuantumDevice, QuantumProvider
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

from metriq_gym.local.device import LocalAerDevice


class LocalProvider(QuantumProvider):
    def __init__(self) -> None:
        super().__init__()
        self.device = LocalAerDevice(provider=self)
        self._devices: dict[str, LocalAerDevice] = {}

    def get_devices(self, **_) -> list[QuantumDevice]:
        devices = [self.device]

        # Try to get all available IBM fake backends.
        try:
            service = QiskitRuntimeService()
            backends = service.backends()
            # Add available backends to the list
            for backend in backends:
                device_id = backend.name
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

    def get_device(self, device_id: str) -> QuantumDevice:
        if device_id in self._devices:
            return self._devices[device_id]

        if device_id == "aer_simulator":
            return self.device
        try:
            # Try loading a local fake backend first, which doesn't require an
            # IBM Quantum account, otherwise load from the runtime service.

            fake_local_backends = {b.name: b for b in FakeProviderForBackendV2().backends()}
            if device_id in fake_local_backends:
                backend = fake_local_backends[device_id]
            else:
                backend = QiskitRuntimeService().backend(device_id)
            aer_backend = AerSimulator.from_backend(backend)
        except Exception as exc:
            raise ValueError("Unknown device identifier") from exc

        device = LocalAerDevice(provider=self, device_id=device_id, backend=aer_backend)
        self._devices[device_id] = device
        return device
