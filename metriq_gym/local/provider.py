from qbraid.runtime import QuantumProvider
from metriq_gym.local.device import LocalAerDevice


class LocalProvider(QuantumProvider):
    def __init__(self):
        super().__init__()
        self.device = LocalAerDevice(provider=self)

    def get_devices(self, **_):
        return [self.device]

    def get_device(self, device_id, **kwargs):
        if device_id == "aer_simulator":
            return self.device

        elif device_id == "aer_simulator_noisy":
            from qiskit_aer.noise import NoiseModel
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime import QiskitRuntimeService

            noise_backend_name = kwargs.get("noise_backend")
            if not noise_backend_name:
                raise ValueError(
                    "The 'noise_backend' argument is required for 'aer_simulator_noisy'."
                )

            service = QiskitRuntimeService()
            real_backend = service.backend(noise_backend_name)

            noise_model = NoiseModel.from_backend(real_backend)
            coupling_map = real_backend.configuration().coupling_map
            basis_gates = noise_model.basis_gates

            noisy_simulator = AerSimulator(
                noise_model=noise_model,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
            )

            return LocalAerDevice(provider=self, backend=noisy_simulator, device_id="aer_simulator_noisy")

        else:
            raise ValueError(f"Unknown device identifier: {device_id}")
