"""qBraid provider implementation for OriginQ Wukong devices."""

from typing import Any

from qbraid.runtime import QuantumProvider
from .device import OriginDevice, SIMULATOR_BACKENDS
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

    def get_devices(self, *, hardware_only: bool | None = None, **_: Any) -> list[OriginDevice]:
        catalog = self.service.backends()
        device_ids: set[str] = set()
        for backend_id, available in catalog.items():
            if available is False:
                continue
            if hardware_only and backend_id in SIMULATOR_BACKENDS:
                continue
            device_ids.add(backend_id)

        # Return devices sorted for deterministic CLI output
        return [self.get_device(device_id) for device_id in sorted(device_ids)]

    def get_device(self, device_id: str) -> OriginDevice:
        device_id = device_id.strip()
        if device_id not in self._devices:
            backend = self.service.backend(device_id)
            self._devices[device_id] = OriginDevice(
                provider=self,
                device_id=device_id,
                backend=backend,
                backend_name=device_id,
            )
        return self._devices[device_id]
