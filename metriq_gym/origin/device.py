"""qBraid device wrapper for OriginQ Wukong hardware and simulators."""

import logging
from collections.abc import Sequence
from typing import Any

from pyqpanda3.intermediate_compiler import convert_qasm_string_to_qprog
from pyqpanda3.qcloud import QCloudBackend, ChipInfo
from qbraid import QPROGRAM
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import DeviceStatus, QuantumDevice, TargetProfile
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps

from metriq_gym.origin.job import OriginJob
from metriq_gym.origin.qcloud_utils import get_qcloud_options

logger = logging.getLogger(__name__)

SIMULATOR_BACKENDS = {
    "full_amplitude": 35,
    "partial_amplitude": 68,
    "single_amplitude": 200,
}


def get_origin_connectivity(device: "OriginDevice") -> tuple[list[int], list[tuple[int, int]]]:
    """Return active qubits and connectivity edges for an Origin device."""

    try:
        chip_info: ChipInfo = device.backend.chip_info()
    except Exception:  # pragma: no cover - depends on live service
        return [], []
    available_qubits: list[int] = chip_info.available_qubits()
    return (available_qubits, chip_info.get_chip_topology(available_qubits))


def _infer_num_qubits(backend: QCloudBackend, backend_name: str, *, simulator: bool) -> int | None:
    if simulator:
        return SIMULATOR_BACKENDS.get(backend_name)
    total_qubits: int | None = None
    try:
        chip_info = backend.chip_info()
        total_qubits = chip_info.qubits_num()
    except Exception:  # pragma: no cover - defensive programming when API changes
        logger.debug("Unable to determine qubit count from chip info", exc_info=True)
    return total_qubits


def _infer_basis_gates(backend: QCloudBackend, *, simulator: bool) -> list[str] | None:
    if simulator:
        return None
    try:
        chip_info = backend.chip_info()
        return chip_info.get_basic_gates()
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
