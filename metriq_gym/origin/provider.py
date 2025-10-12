"""qBraid provider implementation for OriginQ Wukong devices."""

from typing import Any

from qbraid.runtime import QuantumProvider

from ._constants import ALIAS_TO_DISPLAY, BACKEND_ALIASES, SIMULATOR_BACKENDS
from .device import OriginDevice
from .qcloud_utils import get_service


class OriginProvider(QuantumProvider):
    """Adapter that exposes OriginQ Wukong devices through the qBraid runtime API."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key
        self._service: Any | None = None
        self._devices: dict[str, OriginDevice] = {}

    @property
    def service(self):
        if self._service is None:
            self._service = get_service(explicit_key=self._api_key)
        return self._service

    def _backend_name(self, device_id: str) -> str:
        return BACKEND_ALIASES.get(device_id, device_id)

    def get_devices(self, *, hardware_only: bool | None = None, **_: Any) -> list[OriginDevice]:
        catalog = self.service.backends()
        device_ids: set[str] = set()
        for backend_id, available in catalog.items():
            if available is False:
                continue
            alias = ALIAS_TO_DISPLAY.get(backend_id, backend_id)
            if hardware_only and backend_id in SIMULATOR_BACKENDS:
                continue
            device_ids.add(alias)

        # Ensure we expose a stable alias even if QCloud omits it from the listing.
        if "origin_wukong" not in device_ids:
            target_backend = BACKEND_ALIASES["origin_wukong"]
            if catalog.get(target_backend, True):
                device_ids.add("origin_wukong")

        # Return devices sorted for deterministic CLI output
        return [self.get_device(device_id) for device_id in sorted(device_ids)]

    def get_device(self, device_id: str) -> OriginDevice:
        device_id = device_id.strip()
        if device_id not in self._devices:
            backend_name = self._backend_name(device_id)
            backend = self.service.backend(backend_name)
            self._devices[device_id] = OriginDevice(
                provider=self,
                device_id=device_id,
                backend=backend,
                backend_name=backend_name,
            )
        return self._devices[device_id]
