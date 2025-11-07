"""WIT (Wormhole-inspired teleportation) benchmark for the Metriq Gym
(credit to Paul Nation for the original code for IBM devices).

The WIT benchmark is based on the following paper:
    Towards Quantum Gravity in the Lab on Quantum Processors
    Illya Shapoval, Vincent Paul Su, Wibe de Jong, Miro Urbanek, Brian Swingle
    Quantum 7, 1138 (2023)

A generalized version of the WIT benchmark software can also be found as a companion [software
repository](https://gitlab.com/ishapova/qglab/-/blob/master/scripts/wormhole.py) to the above paper.
"""

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence, SupportsFloat, SupportsInt, cast

from qiskit import QuantumCircuit
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.statistics import (
    binary_expectation_stddev,
    binary_expectation_value,
)

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


VALID_MESSAGE_METHODS = ("reset", "swap", "transfer")


@dataclass(frozen=True)
class WormholeTeleportationConfig:
    """Configuration for the generalized WIT circuit."""

    n_qubits_per_side: int
    message_size: int
    x_rotation_transverse_angle: float
    zz_rotation_angle: float
    z_rotation_angles: Sequence[float]
    time_steps: int
    insert_message_method: str
    interaction_coupling_strength: float

    def __post_init__(self) -> None:
        if self.n_qubits_per_side < 1:
            raise ValueError("n_qubits_per_side must be at least 1.")
        if self.message_size < 1:
            raise ValueError("message_size must be at least 1.")
        if self.message_size >= self.n_qubits_per_side:
            raise ValueError("message_size must be smaller than n_qubits_per_side.")
        if self.time_steps < 1:
            raise ValueError("time_steps must be at least 1.")
        if self.insert_message_method not in VALID_MESSAGE_METHODS:
            raise ValueError(
                f"insert_message_method must be one of {', '.join(VALID_MESSAGE_METHODS)}."
            )
        if len(tuple(self.z_rotation_angles)) != self.n_qubits_per_side:
            raise ValueError("z_rotation_angles length must match n_qubits_per_side.")
        if self.insert_message_method in {"swap", "transfer"} and self.message_size != 1:
            raise ValueError("message_size must be 1 when using swap or transfer insertion.")

    @property
    def total_qubits(self) -> int:
        extra = 1 if self.insert_message_method in {"swap", "transfer"} else 0
        return 2 * self.n_qubits_per_side + extra


class WormholeTeleporterFactory:
    """Helper for constructing holographic teleportation circuits."""

    def __init__(self, config: WormholeTeleportationConfig):
        self.config = config

    def _left_indices(self) -> list[int]:
        return list(range(self.config.n_qubits_per_side))

    def _right_indices(self) -> list[int]:
        start = self.config.n_qubits_per_side
        end = 2 * self.config.n_qubits_per_side
        return list(range(start, end))[::-1]

    @staticmethod
    def _make_bell_pair(qubit_pair: tuple[int, int], circuit: QuantumCircuit) -> None:
        left, right = qubit_pair
        circuit.h(left)
        circuit.cx(left, right)

    def _prepare_tfd_state(
        self, circuit: QuantumCircuit, left: Sequence[int], right: Sequence[int]
    ) -> None:
        for pair in zip(left, right):
            self._make_bell_pair(pair, circuit)

    @staticmethod
    def _apply_transverse_rotation(
        qubits: Iterable[int], circuit: QuantumCircuit, theta: float
    ) -> None:
        for qubit in qubits:
            circuit.rx(2 * theta, qubit)

    @staticmethod
    def _apply_ising_evolution(
        qubits: Sequence[int], circuit: QuantumCircuit, theta: float, phases: Sequence[float]
    ) -> None:
        for qubit, phi in zip(qubits, phases, strict=True):
            circuit.rz(2 * phi, qubit)

        for q1, q2 in zip(qubits, qubits[1:]):
            circuit.rzz(2 * theta, q1, q2)

    def _hamiltonian_step(
        self,
        qubits: Sequence[int],
        circuit: QuantumCircuit,
        *,
        forward: bool,
        transpose: bool,
    ) -> None:
        theta_b = self.config.x_rotation_transverse_angle
        theta_j = self.config.zz_rotation_angle
        phases = list(self.config.z_rotation_angles)

        if not forward:
            theta_b = -theta_b
            theta_j = -theta_j
            phases = [-phi for phi in phases]

        gates_first = transpose != forward
        for _ in range(self.config.time_steps):
            if gates_first:
                self._apply_ising_evolution(qubits, circuit, theta_j, phases)
                self._apply_transverse_rotation(qubits, circuit, theta_b)
            else:
                self._apply_transverse_rotation(qubits, circuit, theta_b)
                self._apply_ising_evolution(qubits, circuit, theta_j, phases)

    def _two_sided_coupling(
        self, circuit: QuantumCircuit, left: Sequence[int], right: Sequence[int]
    ) -> None:
        denom = self.config.n_qubits_per_side - self.config.message_size
        prefactor = 1.0 / denom
        coupling_angle = 2 * self.config.interaction_coupling_strength * prefactor
        for l_qubit, r_qubit in zip(
            left[self.config.message_size :], right[self.config.message_size :], strict=True
        ):
            circuit.rzz(coupling_angle, l_qubit, r_qubit)

    def circuit(self) -> QuantumCircuit:
        total = self.config.total_qubits
        circuit = QuantumCircuit(
            total,
            1,
            name=f"wit_g_{self.config.interaction_coupling_strength:.4f}",
        )

        left_indices = self._left_indices()
        right_indices = self._right_indices()
        ancilla_index: int | None = (
            2 * self.config.n_qubits_per_side
            if self.config.insert_message_method in {"swap", "transfer"}
            else None
        )

        self._prepare_tfd_state(circuit, left_indices, right_indices)
        self._hamiltonian_step(left_indices, circuit, forward=False, transpose=False)

        if self.config.insert_message_method == "swap":
            if ancilla_index is None:
                raise ValueError("swap insertion requires an ancilla qubit.")
            circuit.swap(left_indices[0], ancilla_index)
        elif self.config.insert_message_method == "reset":
            circuit.reset(left_indices[0])
        else:  # transfer
            if ancilla_index is None:
                raise ValueError("transfer insertion requires an ancilla qubit.")
            updated_left = list(left_indices)
            updated_left[0] = ancilla_index
            left_indices = updated_left

        self._hamiltonian_step(left_indices, circuit, forward=True, transpose=False)
        self._two_sided_coupling(circuit, left_indices, right_indices)
        self._hamiltonian_step(right_indices, circuit, forward=True, transpose=True)
        circuit.measure(right_indices[0], 0)
        return circuit


def wit_circuit(config: WormholeTeleportationConfig) -> QuantumCircuit:
    """Build a WIT circuit from the provided configuration."""
    return WormholeTeleporterFactory(config).circuit()


def _legacy_config(total_qubits: int) -> WormholeTeleportationConfig:
    if total_qubits == 6:
        insert_method = "reset"
    elif total_qubits == 7:
        insert_method = "swap"
    else:
        raise ValueError(f"Unsupported legacy WIT qubit count: {total_qubits}.")
    return WormholeTeleportationConfig(
        n_qubits_per_side=3,
        message_size=1,
        x_rotation_transverse_angle=float(np.pi / 4),
        zz_rotation_angle=float(np.pi / 4),
        z_rotation_angles=(0.0283397, 0.00519953, 0.0316079),
        time_steps=3,
        insert_message_method=insert_method,
        interaction_coupling_strength=float(np.pi / 2),
    )


def legacy_wit_circuit(total_qubits: int) -> QuantumCircuit:
    """Return the legacy 6- or 7-qubit WIT circuit."""
    return wit_circuit(_legacy_config(total_qubits))


def build_wit_config_from_params(params) -> WormholeTeleportationConfig:
    """Convert validated schema parameters into a WormholeTeleportationConfig."""
    legacy_total = getattr(params, "num_qubits", None)
    n_qubits_per_side = getattr(params, "n_qubits_per_side", None)

    if n_qubits_per_side is None:
        if legacy_total is None:
            raise ValueError(
                "WIT parameters must include n_qubits_per_side for the generalized circuit."
            )
        config = _legacy_config(int(cast(SupportsInt, legacy_total)))
    else:
        raw_fields = {
            "message_size": getattr(params, "message_size", None),
            "x_rotation_transverse_angle": getattr(params, "x_rotation_transverse_angle", None),
            "zz_rotation_angle": getattr(params, "zz_rotation_angle", None),
            "z_rotation_angles": getattr(params, "z_rotation_angles", None),
            "time_steps": getattr(params, "time_steps", None),
            "insert_message_method": getattr(params, "insert_message_method", None),
            "interaction_coupling_strength": getattr(params, "interaction_coupling_strength", None),
        }

        missing = [name for name, value in raw_fields.items() if value is None]
        if missing:
            raise ValueError("Missing required WIT parameters: " + ", ".join(sorted(missing)))

        message_size = int(cast(SupportsInt, raw_fields["message_size"]))
        x_rotation = float(cast(SupportsFloat, raw_fields["x_rotation_transverse_angle"]))
        zz_rotation = float(cast(SupportsFloat, raw_fields["zz_rotation_angle"]))
        z_rotation_angles_raw = cast(Sequence[SupportsFloat], raw_fields["z_rotation_angles"])
        z_rotation_angles = tuple(float(v) for v in z_rotation_angles_raw)
        time_steps = int(cast(SupportsInt, raw_fields["time_steps"]))
        insert_method_str = str(raw_fields["insert_message_method"]).lower()
        coupling_strength = float(cast(SupportsFloat, raw_fields["interaction_coupling_strength"]))

        n_qubits_per_side_int = int(cast(SupportsInt, n_qubits_per_side))

        config = WormholeTeleportationConfig(
            n_qubits_per_side=n_qubits_per_side_int,
            message_size=message_size,
            x_rotation_transverse_angle=x_rotation,
            zz_rotation_angle=zz_rotation,
            z_rotation_angles=z_rotation_angles,
            time_steps=time_steps,
            insert_message_method=insert_method_str,
            interaction_coupling_strength=coupling_strength,
        )

    if legacy_total is not None and int(cast(SupportsInt, legacy_total)) != config.total_qubits:
        raise ValueError(
            "num_qubits does not match the total qubits implied by the generalized WIT parameters."
        )

    return config


class WITResult(BenchmarkResult):
    expectation_value: BenchmarkScore

    def compute_score(self) -> float | None:
        return self.expectation_value.value


@dataclass
class WITData(BenchmarkData):
    pass


class WIT(Benchmark):
    def dispatch_handler(self, device: "QuantumDevice") -> WITData:
        config = build_wit_config_from_params(self.params)
        shots_value = getattr(self.params, "shots", None)
        if shots_value is None:
            raise ValueError("WIT parameters must include 'shots'.")
        shots = int(cast(SupportsInt, shots_value))
        return WITData.from_quantum_job(device.run(wit_circuit(config), shots=shots))

    def poll_handler(
        self,
        job_data: WITData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> WITResult:
        counts = flatten_counts(result_data)[0]
        shots_value = getattr(self.params, "shots", None)
        if shots_value is None:
            raise ValueError("WIT parameters must include 'shots'.")
        shots = int(cast(SupportsInt, shots_value))
        return WITResult(
            expectation_value=BenchmarkScore(
                value=binary_expectation_value(shots, counts),
                uncertainty=binary_expectation_stddev(shots, counts),
            )
        )
