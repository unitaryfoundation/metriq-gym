from qbraid.runtime import QuantumDevice, QuantumProvider
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

from metriq_gym.local.device import LocalAerDevice


def _normalize_device_id(device_id: str) -> str:
    return device_id.lower().replace("-", "_")


def _backend_name(backend) -> str:
    name = getattr(backend, "name", "")
    if callable(name):
        name = name()
    return str(name)


def _fake_backend_aliases(backend) -> set[str]:
    name = _normalize_device_id(_backend_name(backend))
    if not name:
        return set()

    aliases = {name}
    if name.startswith("fake_"):
        suffix = name.removeprefix("fake_")
        aliases.update({suffix, f"ibm_{suffix}"})

    return aliases


class LocalProvider(QuantumProvider):
    def __init__(self) -> None:
        super().__init__()
        self.device = LocalAerDevice(provider=self)
        self._devices: dict[str, LocalAerDevice] = {}
        self._fake_local_backends: dict[str, object] | None = None

    def _get_fake_local_backends(self) -> dict[str, object]:
        if self._fake_local_backends is None:
            self._fake_local_backends = {}
            for backend in FakeProviderForBackendV2().backends():
                for alias in _fake_backend_aliases(backend):
                    self._fake_local_backends[alias] = backend

        return self._fake_local_backends

    def get_devices(self, **_) -> list[QuantumDevice]:
        devices = [self.device]

        # Try to get all available IBM runtime backends.
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

            fake_local_backends = self._get_fake_local_backends()
            normalized_device_id = _normalize_device_id(device_id)
            if normalized_device_id in fake_local_backends:
                backend = fake_local_backends[normalized_device_id]
            else:
                backend = QiskitRuntimeService().backend(device_id)
            aer_backend = AerSimulator.from_backend(backend)
        except Exception as exc:
            raise ValueError("Unknown device identifier") from exc

        device = LocalAerDevice(provider=self, device_id=device_id, backend=aer_backend)
        self._devices[device_id] = device
        return device
