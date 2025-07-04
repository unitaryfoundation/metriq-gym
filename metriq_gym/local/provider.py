from qbraid.runtime import QuantumProvider
from metriq_gym.local.device import LocalAerDevice


class LocalProvider(QuantumProvider):
    def __init__(self):
        super().__init__()
        self.device = LocalAerDevice(provider=self)

    def get_devices(self, **_):
        return [self.device]

    def get_device(self, device_id):
        if device_id == "aer_simulator":
            return self.device
        else:
            raise ValueError("Unknown device identifier")
