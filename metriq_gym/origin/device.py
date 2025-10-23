"""qBraid device wrapper for OriginQ Wukong hardware and simulators."""

import logging
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from pyqpanda3.intermediate_compiler import convert_qasm_string_to_qprog
from qbraid import QPROGRAM
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import DeviceStatus, QuantumDevice, TargetProfile
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps

from ._constants import SIMULATOR_BACKENDS, SIMULATOR_MAX_QUBITS
from .job import OriginJob
from .qcloud_utils import get_qcloud_options

if TYPE_CHECKING:  # pragma: no cover
    from pyqpanda3.qcloud import QCloudBackend

logger = logging.getLogger(__name__)


def _sanitize_int_list(values: Any) -> list[int]:
    """Return a sorted list of unique integers from an iterable."""

    sanitized: set[int] = set()
    if not values:
        return []
    for value in values:
        try:
            sanitized.add(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(sanitized)


def _sanitize_edges(values: Any) -> list[tuple[int, int]]:
    """Return sorted, deduplicated undirected edges from a raw topology list."""

    unique_edges: set[tuple[int, int]] = set()
    if not values:
        return []
    for edge in values:
        if not edge or len(edge) < 2:
            continue
        try:
            a, b = int(edge[0]), int(edge[1])
        except (TypeError, ValueError):
            continue
        if a == b:
            continue
        unique_edges.add((min(a, b), max(a, b)))
    return sorted(unique_edges)


def _chip_active_qubits(chip_info: Any) -> list[int]:
    """Return the calibrated qubits reported by the chip."""

    for attr in ("high_frequency_qubits", "available_qubits"):
        try:
            values = getattr(chip_info, attr)()
        except Exception:  # pragma: no cover - depends on live service
            continue
        qubits = _sanitize_int_list(values)
        if qubits:
            return qubits
    return []


def _chip_topology_edges(chip_info: Any) -> list[tuple[int, int]]:
    """Return the undirected edges describing chip connectivity."""

    raw_edges: list[Any]
    try:
        raw_edges = getattr(chip_info, "get_chip_topology", lambda: [])()
    except Exception:  # pragma: no cover - depends on live service
        raw_edges = []
    edges = _sanitize_edges(raw_edges)
    if edges:
        return edges

    try:
        double_infos = chip_info.double_qubits_info()
    except Exception:  # pragma: no cover - depends on live service
        double_infos = None
    if not double_infos:
        return []

    pairs: set[tuple[int, int]] = set()
    for info in double_infos:
        try:
            qubits = info.get_qubits()
        except Exception:  # pragma: no cover - defensive API handling
            continue
        if not qubits or len(qubits) < 2:
            continue
        try:
            a, b = int(qubits[0]), int(qubits[1])
        except (TypeError, ValueError):
            continue
        if a == b:
            continue
        pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def get_origin_connectivity(device: "OriginDevice") -> tuple[list[int], list[tuple[int, int]]]:
    """Return active qubits and connectivity edges for an Origin device."""

    try:
        chip_info = device.backend.chip_info()
    except Exception:  # pragma: no cover - depends on live service
        return [], []

    active = _chip_active_qubits(chip_info)
    edges = _chip_topology_edges(chip_info)
    if not active and edges:
        active = sorted({node for edge in edges for node in edge})
    return active, edges


def _infer_num_qubits(
    backend: "QCloudBackend", backend_name: str, *, simulator: bool
) -> int | None:
    if simulator:
        return SIMULATOR_MAX_QUBITS.get(backend_name)
    try:
        chip_info = backend.chip_info()
    except Exception:  # pragma: no cover - depends on live service
        return None
    active_qubits = _chip_active_qubits(chip_info)
    if active_qubits:
        return len(active_qubits)
    try:
        return int(chip_info.qubits_num())
    except Exception:  # pragma: no cover - defensive programming when API changes
        logger.debug("Unable to determine qubit count from chip info", exc_info=True)
        return None


def _infer_basis_gates(backend: "QCloudBackend", *, simulator: bool) -> list[str] | None:
    if simulator:
        return None
    try:
        chip_info = backend.chip_info()
        gates = chip_info.get_basic_gates()
        return list(gates) if gates else None
    except Exception:  # pragma: no cover - depends on live service
        logger.debug("Unable to determine basis gates from chip info", exc_info=True)
        return None


class OriginDevice(QuantumDevice):
    """Concrete qBraid device backed by an OriginQ QCloud backend."""

    def __init__(
        self,
        *,
        provider,
        device_id: str,
        backend: "QCloudBackend",
        backend_name: str,
    ) -> None:
        simulator = backend_name in SIMULATOR_BACKENDS
        profile = TargetProfile(
            device_id=device_id,
            simulator=simulator,
            experiment_type=ExperimentType.GATE_MODEL,
            num_qubits=_infer_num_qubits(backend, backend_name, simulator=simulator),
            program_spec=ProgramSpec(QuantumCircuit),
            basis_gates=_infer_basis_gates(backend, simulator=simulator),
            provider_name="origin",
            extra={"backend_name": backend_name},
        )
        super().__init__(profile)
        self._provider = provider
        self._backend = backend
        self._backend_name = backend_name

    @property
    def provider(self):
        return self._provider

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def backend(self) -> Any:
        return self._backend

    def status(self) -> DeviceStatus:
        # The service does not currently expose health endpoints; assume reachable when instantiated.
        return DeviceStatus.ONLINE

    def _circuit_to_qprog(self, circuit: QuantumCircuit):
        qasm = qasm2_dumps(circuit)
        return convert_qasm_string_to_qprog(qasm)

    def _to_qprog(self, run_input: QPROGRAM):
        if isinstance(run_input, QuantumCircuit):
            return self._circuit_to_qprog(run_input)
        if isinstance(run_input, str):
            # Assume OpenQASM provided by caller
            return convert_qasm_string_to_qprog(run_input)
        raise TypeError(
            f"Unsupported run_input type {type(run_input)}; expected QuantumCircuit or OpenQASM string"
        )

    def transform(self, run_input: QPROGRAM | Sequence[QPROGRAM]):
        if isinstance(run_input, Sequence) and not isinstance(run_input, (QuantumCircuit, str)):
            return [self._to_qprog(item) for item in run_input]
        return [self._to_qprog(run_input)]

    def run(self, run_input: QPROGRAM | Sequence[QPROGRAM], *args, **kwargs):
        if isinstance(run_input, Sequence) and not isinstance(run_input, (QuantumCircuit, str)):
            return [self.submit(item, *args, **kwargs) for item in run_input]
        return self.submit(run_input, *args, **kwargs)

    def submit(self, run_input: QPROGRAM, *, shots: int, **_: Any) -> OriginJob:
        qprog = self._to_qprog(run_input)
        nshots = int(shots)
        if self._backend_name in SIMULATOR_BACKENDS:
            job = self._backend.run(qprog, nshots)
        else:
            options = get_qcloud_options()
            job = self._backend.run(qprog, nshots, options)
        job_id = job.job_id()
        return OriginJob(job_id, device=self, backend_job=job)
