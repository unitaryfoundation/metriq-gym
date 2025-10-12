"""qBraid provider implementation for OriginQ Wukong devices."""

from typing import Any

from qbraid.runtime import QuantumProvider

from ._constants import (
    PRIMARY_BACKEND_ALIAS,
    PRIMARY_BACKEND_ID,
    SIMULATOR_BACKENDS,
)
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
        if device_id == PRIMARY_BACKEND_ALIAS:
            return PRIMARY_BACKEND_ID
        return device_id

    @staticmethod
    def _alias_for_backend(backend_id: str) -> str:
        if backend_id == PRIMARY_BACKEND_ID:
            return PRIMARY_BACKEND_ALIAS
        return backend_id

    def get_devices(self, *, hardware_only: bool | None = None, **_: Any) -> list[OriginDevice]:
        catalog = self.service.backends()
        device_ids: set[str] = set()
        for backend_id, available in catalog.items():
            if available is False:
                continue
            alias = self._alias_for_backend(backend_id)
            if hardware_only and backend_id in SIMULATOR_BACKENDS:
                continue
            device_ids.add(alias)

        # Ensure we expose a stable alias even if QCloud omits it from the listing.
        if PRIMARY_BACKEND_ALIAS not in device_ids:
            if catalog.get(PRIMARY_BACKEND_ID, True):
                device_ids.add(PRIMARY_BACKEND_ALIAS)

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
