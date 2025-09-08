from __future__ import annotations

from typing import Any

from qbraid.runtime import QuantumProvider

from .device import QuantinuumDevice


class QuantinuumProvider(QuantumProvider):
    """qBraid provider for Quantinuum NEXUS targets.

    Advertises a subset of known targets including emulators and syntax
    checkers (e.g., H1-1E, H1-2E, H1-1SC, H1-2SC). No local fallback is
    provided; submission/validation requires a Quantinuum account via
    pytket-quantinuum. Device discovery attempts to list accessible
    targets when credentials are present.
    """

    def __init__(self) -> None:
        super().__init__()
        self._devices: dict[str, QuantinuumDevice] = {}

    def get_devices(self, **_: Any) -> list[QuantinuumDevice]:
        # If credentials and pytket are available, list accessible devices
        devices: list[QuantinuumDevice] = []
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
            from .auth import load_api

            api = load_api()
            avail = getattr(QuantinuumBackend, "available_devices", None)
            if callable(avail):
                try:
                    res = avail(api_handler=api)
                except TypeError:
                    res = avail()
                for item in res or []:
                    name: str | None = None
                    n_qubits: int | None = None
                    if isinstance(item, str):
                        name = item
                    elif isinstance(item, dict):
                        name = item.get("name") or item.get("device_name") or item.get("label")
                        n_qubits = item.get("n_qubits") or item.get("num_qubits")
                    if name:
                        devices.append(QuantinuumDevice(provider=self, device_id=name, num_qubits=n_qubits))
        except Exception:
            # If discovery fails (e.g., no creds), return empty list; get_device will still work lazily
            pass
        return devices

    def get_device(self, device_id: str) -> QuantinuumDevice:
        if device_id in self._devices:
            return self._devices[device_id]

        # Lazily create device without pre-known metadata; operations will fail only when used without access
        dev = QuantinuumDevice(provider=self, device_id=device_id)
        self._devices[device_id] = dev
        return dev
