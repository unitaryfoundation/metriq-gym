"""Utilities for estimating resource requirements of metriq-gym benchmarks."""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


from qbraid import QuantumDevice
from qiskit import QuantumCircuit
from tabulate import tabulate

from metriq_gym.constants import JobType


@dataclass
class GateCounts:
    one_qubit: int = 0
    two_qubit: int = 0
    multi_qubit: int = 0
    measurements: int = 0
    resets: int = 0

    def add(self, other: "GateCounts") -> None:
        self.one_qubit += other.one_qubit
        self.two_qubit += other.two_qubit
        self.multi_qubit += other.multi_qubit
        self.measurements += other.measurements
        self.resets += other.resets


@dataclass
class CircuitEstimate:
    job_index: int
    circuit_index: int
    qubit_count: int
    shots: int
    gate_counts: GateCounts
    depth: int
    hqc: float | None


@dataclass
class ResourceEstimate:
    job_count: int
    circuit_count: int
    total_shots: int
    max_qubits: int
    total_gate_counts: GateCounts
    hqc_total: float | None
    per_circuit: list[CircuitEstimate] = field(default_factory=list)


@dataclass
class CircuitBatch:
    circuits: list[QuantumCircuit]
    shots: int


def _count_gates(circuit: QuantumCircuit) -> GateCounts:
    counts = GateCounts()
    for inst in circuit.data:
        instruction = inst.operation
        qargs = inst.qubits
        cargs = inst.clbits
        name = instruction.name
        num_qubits = len(qargs)
        if name == "measure":
            counts.measurements += len(cargs) or num_qubits
        elif name == "reset":
            counts.resets += num_qubits
        else:
            if num_qubits == 1:
                counts.one_qubit += 1
            elif num_qubits == 2:
                counts.two_qubit += 1
            else:
                counts.multi_qubit += 1
    return counts


HQCFunction = Callable[[GateCounts, int], float]


def _require_device(device: QuantumDevice | None, benchmark: str) -> QuantumDevice:
    if device is None:
        raise ValueError(f"{benchmark} benchmark requires a device to estimate resources.")
    return device


def aggregate_resource_estimates(
    batches: Iterable[CircuitBatch],
    hqc_fn: HQCFunction | None = None,
) -> ResourceEstimate:
    job_count = 0
    circuit_count = 0
    total_shots = 0
    max_qubits = 0
    total_counts = GateCounts()
    per_circuit: list[CircuitEstimate] = []
    hqc_total: float | None = None

    for job_index, batch in enumerate(batches):
        job_count += 1
        for circuit_index, circuit in enumerate(batch.circuits):
            circuit_count += 1
            shots = int(batch.shots)
            gate_counts = _count_gates(circuit)
            total_counts.add(gate_counts)
            total_shots += shots
            max_qubits = max(max_qubits, circuit.num_qubits)

            circuit_hqc: float | None = None
            if hqc_fn is not None:
                circuit_hqc = hqc_fn(gate_counts, shots)
                if hqc_total is None:
                    hqc_total = 0.0
                hqc_total += circuit_hqc
            per_circuit.append(
                CircuitEstimate(
                    job_index=job_index,
                    circuit_index=circuit_index,
                    qubit_count=circuit.num_qubits,
                    shots=shots,
                    gate_counts=gate_counts,
                    depth=circuit.depth(),
                    hqc=circuit_hqc,
                )
            )

    return ResourceEstimate(
        job_count=job_count,
        circuit_count=circuit_count,
        total_shots=total_shots,
        max_qubits=max_qubits,
        total_gate_counts=total_counts,
        hqc_total=hqc_total,
        per_circuit=per_circuit,
    )


def quantinuum_hqc_formula(counts: GateCounts, shots: int) -> float:
    """Compute Quantinuum HQCs using the published formula."""

    n_one = counts.one_qubit
    n_two = counts.two_qubit
    # Includes the implicit initial state preparation (+1) and any resets.
    n_measure = counts.measurements + counts.resets + 1

    gate_term = n_one + 10 * n_two + 5 * n_measure
    return 5.0 + shots * gate_term / 5000.0


def _stat_tuple(values: Sequence[int | float]) -> tuple[str, str, str]:
    if not values:
        return ("0", "0", "0")

    minimum = min(values)
    maximum = max(values)
    average = sum(values) / len(values)

    def fmt(val: float) -> str:
        if abs(val - round(val)) < 1e-9:
            return format(int(round(val)), ",").replace(",", "_")
        return f"{val:.2f}"

    return (fmt(minimum), fmt(maximum), fmt(average))


def print_resource_estimate(
    job_type: JobType, provider: str, device_id: str | None, estimate: ResourceEstimate
) -> None:
    device_label = device_id if device_id else "(no device)"
    print(f"Resource estimate for {job_type.value} on {provider}:{device_label}\n")

    def fmt_int(value: int) -> str:
        return format(value, ",").replace(",", "_")

    summary_rows = [
        ("Jobs", fmt_int(estimate.job_count)),
        ("Circuits", fmt_int(estimate.circuit_count)),
        ("Total shots", fmt_int(estimate.total_shots)),
        ("Max qubits", fmt_int(estimate.max_qubits)),
        ("Total 2q gates", fmt_int(estimate.total_gate_counts.two_qubit)),
        ("Total 1q gates", fmt_int(estimate.total_gate_counts.one_qubit)),
        ("Total multi-qubit gates", fmt_int(estimate.total_gate_counts.multi_qubit)),
        ("Total measurements", fmt_int(estimate.total_gate_counts.measurements)),
        ("Total resets", fmt_int(estimate.total_gate_counts.resets)),
    ]

    if estimate.hqc_total is not None:
        summary_rows.append(("Total HQCs", f"{estimate.hqc_total:,.2f}"))
    else:
        summary_rows.append(("Total HQCs", "n/a"))

    print(tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="github"))

    if not estimate.per_circuit:
        return

    shots = [c.shots for c in estimate.per_circuit]
    one_q = [c.gate_counts.one_qubit for c in estimate.per_circuit]
    two_q = [c.gate_counts.two_qubit for c in estimate.per_circuit]
    multi_q = [c.gate_counts.multi_qubit for c in estimate.per_circuit]
    meas = [c.gate_counts.measurements for c in estimate.per_circuit]
    resets = [c.gate_counts.resets for c in estimate.per_circuit]
    depths = [c.depth for c in estimate.per_circuit]

    stats_rows = [
        ("Shots per circuit", *_stat_tuple(shots)),
        ("2q gates per circuit", *_stat_tuple(two_q)),
        ("1q gates per circuit", *_stat_tuple(one_q)),
        ("Multi-qubit gates per circuit", *_stat_tuple(multi_q)),
        ("Measurements per circuit", *_stat_tuple(meas)),
        ("Resets per circuit", *_stat_tuple(resets)),
        ("Circuit depth", *_stat_tuple(depths)),
    ]

    hqc_values = [c.hqc for c in estimate.per_circuit if c.hqc is not None]
    if hqc_values:
        stats_rows.append(("HQC per circuit", *_stat_tuple(hqc_values)))

    print()
    print(
        tabulate(
            stats_rows,
            headers=["Per-circuit metric", "Min", "Max", "Avg"],
            tablefmt="github",
        )
    )
