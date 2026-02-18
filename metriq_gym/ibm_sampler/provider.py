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
    """IBM provider whose devices support parameterized and twirling via an optional Session."""

    def __init__(
        self,
        token: str | None = None,
        instance: str | None = None,
        channel: ChannelType | None = None,
        **kwargs,
    ):
        """Initialize the provider with IBM Quantum credentials.

        Args:
            token: IBM Quantum API token. If not provided, will attempt to find a saved token
            instance: IBM Quantum instance name. If not provided, will use the default instance
            channel: IBM Quantum channel type (e.g. 'ibm_quantum', 'ibm_cloud'). If not provided, will use the default channel
            **kwargs: Additional keyword arguments to pass to the QiskitRuntimeService constructor
        """
        super().__init__(token=token, instance=instance, channel=channel, **kwargs)

    @cached_method
    def get_devices(self, operational=True, **kwargs) -> list[IBMSamplerDevice]:
        """Get a list of available devices that support SamplerV2 submission.
        Args:
            operational: If True, only return devices that are currently operational. Default is True.
            **kwargs: Additional keyword arguments to filter backends (e.g. n_qubits=5)
        Returns:
            A list of IBMSamplerDevice instances representing the available devices.
        """
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
        """Get a specific device by its ID.
        Args:
            device_id: The ID of the device to retrieve.
            instance: Optional instance name to use when retrieving the device. If not provided, will use the default instance.
        Returns:
            An IBMSamplerDevice instance representing the requested device.
        """
        backend = self.runtime_service.backend(device_id, instance=instance)
        return IBMSamplerDevice(
            profile=self._build_runtime_profile(backend),
            service=self.runtime_service,
        )
