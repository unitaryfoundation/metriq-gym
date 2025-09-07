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
        # If credentials and pytket are available, list accessible devices
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
            from .auth import load_api

            api = load_api()
            avail = getattr(QuantinuumBackend, "available_devices", None)
            names: list[str] = []
            if callable(avail):
                try:
                    res = avail(api_handler=api)
                except TypeError:
                    res = avail()
                for item in res or []:
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict):
                        nm = item.get("name") or item.get("device_name") or item.get("label")
                        if nm:
                            names.append(nm)
            # Filter to supported emulator set if present
            candidates = [n for n in names if n in SUPPORTED_EMULATORS] or list(
                SUPPORTED_EMULATORS.keys()
            )
            return [self.get_device(did) for did in candidates]
        except Exception:
            # Fallback to known emulator identifiers
            return [self.get_device(did) for did in SUPPORTED_EMULATORS.keys()]

    def get_device(self, device_id: str) -> QuantinuumDevice:
        if device_id in self._devices:
            return self._devices[device_id]

        if device_id not in SUPPORTED_EMULATORS:
            raise ValueError("Unknown device identifier")

        dev = QuantinuumDevice(provider=self, device_id=device_id)
        self._devices[device_id] = dev
        return dev
