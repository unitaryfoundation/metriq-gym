import logging
from functools import cache
from typing import Any

from qbraid import QbraidError
from qbraid.runtime import QuantumDevice, QuantumProvider
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

from metriq_gym.local.device import LocalAerDevice


logger = logging.getLogger(__name__)

_RUNTIME_LOOKUP_ERRORS = (QiskitError, ValueError, OSError)
_DEVICE_PROFILE_ERRORS = (AttributeError, TypeError, ValueError, QiskitError)


class UnknownLocalDeviceError(QbraidError, ValueError):
    """Raised when a local device identifier cannot be resolved."""


@cache
def _aer_custom_instructions() -> tuple[str, ...]:
    return tuple(
        _as_string_list(getattr(AerSimulator().configuration(), "custom_instructions", []))
    )


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple | set | frozenset):
        return []
    return [item for item in value if isinstance(item, str)]


def _backend_name(backend: BackendV2) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        name = name()
    if not isinstance(name, str):
        raise TypeError("Backend does not expose a string name")
    return name


def _aer_profile_basis_gates(operation_names: list[str]) -> list[str]:
    custom_instructions = list(_aer_custom_instructions())
    # Mirrors qiskit-aer 0.17.x public from_backend configuration behavior:
    # BackendV2 operation names feed AerBackendConfiguration, then Aer exposes
    # physical basis gates followed by public custom instructions.
    physical_gates = sorted(set(operation_names) - set(custom_instructions) - {"measure"})
    return physical_gates + custom_instructions


def _backend_operation_names(backend: BackendV2) -> list[str] | None:
    operation_names = getattr(backend, "operation_names", None)
    if isinstance(operation_names, list | tuple) and all(
        isinstance(name, str) for name in operation_names
    ):
        return list(operation_names)
    return None


def _backend_coupling_map(backend: BackendV2) -> list[tuple[int, int]] | None:
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is None:
        return None

    get_edges = getattr(coupling_map, "get_edges", None)
    if callable(get_edges):
        return list(get_edges())

    if isinstance(coupling_map, list):
        return coupling_map

    return None


def _backend_max_circuits(backend: BackendV2) -> int | None:
    max_circuits = getattr(backend, "max_circuits", None)
    return max_circuits if isinstance(max_circuits, int) else None


def _make_lazy_configuration(backend: BackendV2) -> Any:
    operation_names = _backend_operation_names(backend)
    backend_version = getattr(backend, "backend_version", None)
    num_qubits = getattr(backend, "num_qubits", None)

    if operation_names is None or not isinstance(num_qubits, int):
        raise TypeError("BackendV2 is missing operation names or qubit count")
    if not isinstance(backend_version, str):
        backend_version = "0.0.0"

    backend_name = _backend_name(backend)

    description = getattr(backend, "description", None)
    if not isinstance(description, str):
        description = "created by AerSimulator.from_backend"

    return AerBackendConfiguration(
        backend_name=f"aer_simulator_from({backend_name})",
        backend_version=backend_version,
        n_qubits=num_qubits,
        basis_gates=_aer_profile_basis_gates(operation_names),
        gates=[],
        max_shots=int(1e6),
        coupling_map=_backend_coupling_map(backend),
        max_experiments=_backend_max_circuits(backend),
        description=description,
    )


class _LazyAerBackend:
    def __init__(self, backend: BackendV2) -> None:
        self._backend = backend
        self._aer_backend: AerSimulator | None = None
        self._configuration: Any | None = None

    def _get_aer_backend(self) -> AerSimulator:
        if self._aer_backend is None:
            self._aer_backend = AerSimulator.from_backend(self._backend)
            self._configuration = None
        return self._aer_backend

    def configuration(self) -> Any:
        if self._aer_backend is not None:
            return self._aer_backend.configuration()
        if self._configuration is None:
            try:
                self._configuration = _make_lazy_configuration(self._backend)
            except _DEVICE_PROFILE_ERRORS:
                self._configuration = self._get_aer_backend().configuration()
        return self._configuration

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_aer_backend().run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_aer_backend(), name)


class LocalProvider(QuantumProvider):
    def __init__(self) -> None:
        super().__init__()
        self.device = LocalAerDevice(provider=self)
        self._devices: dict[str, LocalAerDevice] = {}
        self._fake_local_backends: dict[str, BackendV2] | None = None

    @staticmethod
    def _normalize_device_id(device_id: str) -> str:
        return device_id.strip().lower().replace("-", "_")

    @classmethod
    def _runtime_device_id(cls, device_id: str) -> str | None:
        device_id = cls._normalize_device_id(device_id)
        if device_id == "aer_simulator" or device_id.startswith("fake_"):
            return None
        if device_id.startswith("ibm_"):
            return device_id
        return f"ibm_{device_id}"

    @classmethod
    def _fake_device_id(cls, device_id: str) -> str:
        device_id = cls._normalize_device_id(device_id)
        if device_id.startswith("fake_"):
            return device_id
        if device_id.startswith("ibm_"):
            return f"fake_{device_id.removeprefix('ibm_')}"
        return f"fake_{device_id}"

    @staticmethod
    def _load_fake_local_backends() -> dict[str, BackendV2]:
        return {
            LocalProvider._normalize_device_id(_backend_name(backend)): backend
            for backend in FakeProviderForBackendV2().backends()
        }

    def _get_fake_local_backends(self) -> dict[str, BackendV2]:
        if self._fake_local_backends is None:
            self._fake_local_backends = self._load_fake_local_backends()
        return self._fake_local_backends

    def _get_fake_backend(self, device_id: str) -> tuple[str, BackendV2] | None:
        fake_device_id = self._fake_device_id(device_id)
        backend = self._get_fake_local_backends().get(fake_device_id)
        if backend is None:
            return None
        return fake_device_id, backend

    def _make_device(
        self,
        device_id: str,
        backend: BackendV2,
        *,
        cache_key: str | None = None,
        lazy: bool = False,
    ) -> LocalAerDevice:
        normalized_device_id = self._normalize_device_id(device_id)
        normalized_cache_key = self._normalize_device_id(cache_key or device_id)
        if normalized_cache_key in self._devices:
            return self._devices[normalized_cache_key]

        aer_backend = _LazyAerBackend(backend) if lazy else AerSimulator.from_backend(backend)
        device = LocalAerDevice(
            provider=self, device_id=normalized_device_id, backend=aer_backend
        )
        self._devices[normalized_cache_key] = device
        return device

    def _append_device(
        self, devices: list[QuantumDevice], device_id: str, backend: BackendV2
    ) -> None:
        device = self._make_device(device_id, backend, lazy=True)
        if device not in devices:
            devices.append(device)

    def get_devices(self, **_) -> list[QuantumDevice]:
        devices = [self.device]
        runtime_fake_twins: set[str] = set()

        service = None
        try:
            service = QiskitRuntimeService()
        except _RUNTIME_LOOKUP_ERRORS as exc:
            logger.debug("IBM Runtime service is unavailable for local listing.", exc_info=exc)

        if service is not None:
            try:
                runtime_backends = service.backends()
            except _RUNTIME_LOOKUP_ERRORS as exc:
                logger.debug("Could not list IBM Runtime backends.", exc_info=exc)
                runtime_backends = []

            for backend in runtime_backends:
                try:
                    device_id = self._normalize_device_id(_backend_name(backend))
                    self._append_device(devices, device_id, backend)
                    runtime_fake_twins.add(self._fake_device_id(device_id))
                except _DEVICE_PROFILE_ERRORS as exc:
                    logger.debug("Could not add IBM Runtime backend.", exc_info=exc)

        for device_id, backend in self._get_fake_local_backends().items():
            if device_id in runtime_fake_twins:
                continue
            try:
                self._append_device(devices, device_id, backend)
            except _DEVICE_PROFILE_ERRORS as exc:
                logger.debug("Could not add local fake backend %s.", device_id, exc_info=exc)

        return devices

    def get_device(self, device_id: str) -> QuantumDevice:
        device_id = self._normalize_device_id(device_id)

        if device_id == "aer_simulator":
            return self.device

        runtime_device_id = self._runtime_device_id(device_id)
        if runtime_device_id is not None and runtime_device_id in self._devices:
            return self._devices[runtime_device_id]

        if device_id in self._devices:
            return self._devices[device_id]

        fake_cache_key = self._fake_device_id(device_id)
        if fake_cache_key in self._devices:
            return self._devices[fake_cache_key]

        if runtime_device_id is not None:
            try:
                backend = QiskitRuntimeService().backend(runtime_device_id)
                return self._make_device(runtime_device_id, backend)
            except _RUNTIME_LOOKUP_ERRORS as exc:
                logger.debug(
                    "Falling back from IBM Runtime backend %s to local fake backend.",
                    runtime_device_id,
                    exc_info=exc,
                )

        fake_backend_info = self._get_fake_backend(device_id)
        if fake_backend_info is None:
            raise UnknownLocalDeviceError("Unknown device identifier")

        resolved_fake_device_id, fake_backend = fake_backend_info
        display_device_id = runtime_device_id or resolved_fake_device_id
        try:
            return self._make_device(
                display_device_id,
                fake_backend,
                cache_key=resolved_fake_device_id,
            )
        except _DEVICE_PROFILE_ERRORS as exc:
            raise UnknownLocalDeviceError("Unknown device identifier") from exc
