from __future__ import annotations

from typing import Dict, List

from qbraid.runtime import QuantumProvider


class QuantinuumProvider(QuantumProvider):
    """qBraid-compatible provider backed by the qNexus SDK.

    Exposes Quantinuum devices discoverable via the NEXUS platform and
    returns `QuantinuumDevice` instances capable of compiling/executing
    pytket circuits.
    """

    def __init__(self) -> None:
        super().__init__()
        self._devices: Dict[str, QuantinuumDevice] = {}

    def get_devices(self, **_) -> List[QuantinuumDevice]:  # pragma: no cover - network call
        try:
            import qnexus as qnx  # type: ignore
        except Exception as exc:  # ImportError or env issues
            raise RuntimeError(
                "qnexus is required to list Quantinuum devices. Install the 'quantinuum' extras: `poetry install --extras quantinuum` (use Python 3.12)."
            ) from exc

        # Lazy import to avoid import-time dependency issues
        from .device import QuantinuumDevice

        devices_df = qnx.devices.get_all(issuers=["QUANTINUUM"]).df()
        devices: List[QuantinuumDevice] = []
        for _, row in devices_df.iterrows():
            name = str(row.get("name") or row.get("device_name") or "").strip()
            if not name:
                continue
            device = self._devices.get(name)
            if device is None:
                device = QuantinuumDevice(provider=self, device_id=name)
                self._devices[name] = device
            devices.append(device)

        # In case device discovery fails/sparse, return any cached devices too
        return devices or list(self._devices.values())

    def get_device(self, device_id: str) -> QuantinuumDevice:
        if device_id in self._devices:
            return self._devices[device_id]

        # Lazy-init device with provided identifier; runtime checks occur on submit
        from .device import QuantinuumDevice
        device = QuantinuumDevice(provider=self, device_id=device_id)
        self._devices[device_id] = device
        return device
