from __future__ import annotations

from typing import Any

from qbraid.runtime import QuantumProvider

from .device import QuantinuumDevice, SUPPORTED_EMULATORS


class QuantinuumProvider(QuantumProvider):
    """qBraid provider for Quantinuum NEXUS emulator devices.

    This implementation advertises H1-1E and H1-2E.
    No local fallback is provided; submission requires a
    Quantinuum account (via pytket-quantinuum).
    """

    def __init__(self) -> None:
        super().__init__()
        self._devices: dict[str, QuantinuumDevice] = {}

    def get_devices(self, **_: Any) -> list[QuantinuumDevice]:
        # Lazily instantiate known emulator devices
        return [self.get_device(did) for did in SUPPORTED_EMULATORS.keys()]

    def get_device(self, device_id: str) -> QuantinuumDevice:
        if device_id in self._devices:
            return self._devices[device_id]

        if device_id not in SUPPORTED_EMULATORS:
            raise ValueError("Unknown device identifier")

        dev = QuantinuumDevice(provider=self, device_id=device_id)
        self._devices[device_id] = dev
        return dev
