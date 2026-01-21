"""qBraid provider for IQM Resonance using the official iqm-client SDK."""

from __future__ import annotations

import logging
import os
from typing import Any

from qbraid.runtime import QuantumProvider

from metriq_gym.iqm_resonance.device import IQMResonanceDevice

logger = logging.getLogger(__name__)


class IQMResonanceProvider(QuantumProvider):
    """Expose Resonance devices through the qBraid runtime API."""

    def __init__(self, api_token: str | None = None, base_url: str | None = None) -> None:
        super().__init__()
        self._api_token = api_token or os.getenv("RESONANCE_API_TOKEN") or os.getenv("IQM_TOKEN")
        # Default to the public Resonance API root; allow override via env/base_url
        self._base_url = "https://resonance.meetiqm.com/api/v1"  # TODO: verify correct URL
        self._client = None
        self._devices: dict[str, IQMResonanceDevice] = {}

    def _get_client(self):
        if self._client is None:
            try:
                from iqm.iqm_client import IQMClient
            except Exception as exc:  # noqa: BLE001
                raise ImportError(
                    "iqm-client is required for the IQM provider. Install with `pip install iqm-client`."
                ) from exc

            kwargs: dict[str, Any] = {"url": self._base_url}
            if self._api_token:
                kwargs["token"] = self._api_token
            self._client = IQMClient(**kwargs)
        return self._client

    def get_devices(self, **_: Any) -> list[IQMResonanceDevice]:
        client = self._get_client()
        devices: list[IQMResonanceDevice] = []
        try:
            arch = client.get_quantum_architecture().quantum_architecture
            device_id = arch.name or "iqm"
            if device_id not in self._devices:
                self._devices[device_id] = IQMResonanceDevice(
                    provider=self, device_id=device_id, client=client, architecture=arch
                )
            devices.append(self._devices[device_id])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch Resonance devices: %s", exc)
        return devices

    def get_device(self, device_id: str) -> IQMResonanceDevice:
        device_id = device_id.strip()
        if device_id not in self._devices:
            devices = self.get_devices()
            for dev in devices:
                if dev.id == device_id:
                    return dev
            raise ValueError(f"Device '{device_id}' not found on IQM Resonance")
        return self._devices[device_id]
