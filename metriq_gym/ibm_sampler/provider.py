"""Provider that returns :class:`IBMSamplerDevice` instances.

Delegates all credential handling and backend discovery to qBraid's
``QiskitRuntimeProvider``, but wraps returned devices in
``IBMSamplerDevice`` so that ``submit()`` calls can use parameterized circuits
and twirling options via the SamplerV2 interface.
"""

from qiskit_ibm_runtime.accounts import ChannelType
from qbraid._caching import cached_method
from qbraid.runtime.ibm.provider import QiskitRuntimeProvider
from .device import IBMSamplerDevice


class IBMSamplerProvider(QiskitRuntimeProvider):
    """IBM provider whose devices always submit via a Session."""

    def __init__(
        self,
        token: str | None = None,
        instance: str | None = None,
        channel: ChannelType | None = None,
        **kwargs,
    ):
        super().__init__(token=token, instance=instance, channel=channel, **kwargs)

    @cached_method
    def get_devices(self, operational=True, **kwargs) -> list[IBMSamplerDevice]:
        backends = self.runtime_service.backends(operational=operational, **kwargs)
        return [
            IBMSamplerDevice(
                profile=self._build_runtime_profile(backend),
                service=self.runtime_service,
            )
            for backend in backends
        ]

    @cached_method
    def get_device(self, device_id: str, instance: str | None = None) -> IBMSamplerDevice:
        backend = self.runtime_service.backend(device_id, instance=instance)
        return IBMSamplerDevice(
            profile=self._build_runtime_profile(backend),
            service=self.runtime_service,
        )
