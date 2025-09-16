import qnexus as qnx
from qbraid.runtime import QuantumProvider

from metriq_gym.quantinuum.device import QuantinuumDevice


class QuantinuumProvider(QuantumProvider):
    def __init__(self) -> None:
        super().__init__()
        self._devices: dict[str, QuantinuumDevice] = {}

    def get_devices(self, **_) -> list[QuantinuumDevice]:
        df = qnx.devices.get_all(issuers=["QUANTINUUM"]).df()
        return [
            self.get_device(str(row.get("name") or row.get("device_name") or "").strip())
            for _, row in df.iterrows()
        ]

    def get_device(self, device_id: str) -> QuantinuumDevice:
        device_id = device_id.strip()
        if device_id not in self._devices:
            self._devices[device_id] = QuantinuumDevice(provider=self, device_id=device_id)
        return self._devices[device_id]
