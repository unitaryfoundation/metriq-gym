"""Utilities for estimating resource requirements of metriq-gym benchmarks."""

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence


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
    ibm_quantum_seconds: float | None = None
    ibm_timing_complete: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class CircuitBatch:
    # TODO change Qiskit QuantumCircuit to qBraid QProgram
    circuits: list[QuantumCircuit]
    shots: int
    # Direct Runtime submissions already use physical circuits and bypass qBraid's transform.
    transpile: bool = True
    notes: list[str] = field(default_factory=list)


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


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Number of two-qubit gates in a circuit.

    Barriers, measurements, and resets are skipped by name so they don't inflate
    the count (a barrier can span two qubits but isn't a gate).
    """
    return sum(
        1
        for inst in circuit.data
        if inst.operation.name not in ("barrier", "measure", "reset") and len(inst.qubits) == 2
    )


HQCFunction = Callable[[GateCounts, int, int], float]


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
    notes: list[str] = []

    for job_index, batch in enumerate(batches):
        notes.extend(note for note in batch.notes if note not in notes)
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
                circuit_hqc = hqc_fn(gate_counts, shots, circuit.num_qubits)
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
        notes=notes,
    )


def quantinuum_hqc_formula(counts: GateCounts, shots: int, num_qubits: int) -> float:
    """Compute Quantinuum HQCs using the published formula.

    HQC formula definition from:
    https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/workflow.html?_gl=1*11onrsb*_gcl_au*MTExNzY5NDM0Mi4xNzYzNTA1MTM3#tracking-usage-with-hardware-quantum-credits-hqcs
    """

    n_one = counts.one_qubit
    n_two = counts.two_qubit
    n_measure = counts.measurements + counts.resets + num_qubits

    gate_term = n_one + 10 * n_two + 5 * n_measure
    return 5.0 + shots * gate_term / 5000.0


def merge_resource_estimates(estimates: Iterable[ResourceEstimate]) -> ResourceEstimate:
    """Combine benchmark estimates without retaining their generated circuits."""
    total = ResourceEstimate(
        job_count=0,
        circuit_count=0,
        total_shots=0,
        max_qubits=0,
        total_gate_counts=GateCounts(),
        hqc_total=None,
    )
    for estimate in estimates:
        total.per_circuit.extend(
            replace(circuit, job_index=circuit.job_index + total.job_count)
            for circuit in estimate.per_circuit
        )
        total.job_count += estimate.job_count
        total.circuit_count += estimate.circuit_count
        total.total_shots += estimate.total_shots
        total.max_qubits = max(total.max_qubits, estimate.max_qubits)
        total.total_gate_counts.add(estimate.total_gate_counts)
        if estimate.hqc_total is not None:
            total.hqc_total = (total.hqc_total or 0.0) + estimate.hqc_total
        if estimate.ibm_quantum_seconds is not None:
            total.ibm_quantum_seconds = (
                total.ibm_quantum_seconds or 0.0
            ) + estimate.ibm_quantum_seconds
        total.ibm_timing_complete &= estimate.ibm_timing_complete
        total.notes.extend(note for note in estimate.notes if note not in total.notes)
    return total


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
    job_type: JobType | str,
    provider: str,
    device_id: str | None,
    estimate: ResourceEstimate,
    *,
    verbose: bool = False,
) -> None:
    provider_key = provider.strip().lower()
    seconds = estimate.ibm_quantum_seconds
    time_label = (
        "Partial IBM QPU time"
        if seconds is not None and not estimate.ibm_timing_complete
        else "Estimated IBM QPU time"
    )
    time_value = f"{seconds:,.3f}" if seconds is not None else "n/a"
    hqc_value = f"{estimate.hqc_total:,.2f}" if estimate.hqc_total is not None else "n/a"
    if not verbose:
        if provider_key == "ibm":
            detail_hint = (
                " (use --verbose for details)"
                if seconds is None or not estimate.ibm_timing_complete
                else ""
            )
            print(f"{time_label} (s): {time_value}{detail_hint}")
        elif provider_key == "quantinuum":
            print(f"Total HQCs: {hqc_value}")
        else:
            print(f"No cost estimate for {provider}. Use --verbose for circuit resources.")
        return

    device_label = device_id if device_id else "(no device)"
    print(f"Resource estimate for {job_type} on {provider}:{device_label}\n")

    def fmt_int(value: int) -> str:
        return format(value, ",").replace(",", "_")

    summary_rows = [
        ("Jobs", fmt_int(estimate.job_count)),
        ("Circuits", fmt_int(estimate.circuit_count)),
        ("Total shots", fmt_int(estimate.total_shots)),
        ("Max qubits", fmt_int(estimate.max_qubits)),
        ("Total 1q gates", fmt_int(estimate.total_gate_counts.one_qubit)),
        ("Total 2q gates", fmt_int(estimate.total_gate_counts.two_qubit)),
        ("Total 3+ qubits gates", fmt_int(estimate.total_gate_counts.multi_qubit)),
        ("Total measurements", fmt_int(estimate.total_gate_counts.measurements)),
        ("Total resets", fmt_int(estimate.total_gate_counts.resets)),
    ]

    if provider_key == "quantinuum":
        summary_rows.append(("Total HQCs", hqc_value))
    elif provider_key == "ibm":
        summary_rows.append((f"{time_label} (s)", time_value))

    print(tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="github"))
    for note in estimate.notes:
        print(f"Note: {note}")
    if provider_key == "ibm" and not estimate.ibm_timing_complete:
        print("IBM QPU timing is incomplete; any subtotal excludes entries without timing.")

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
    if provider_key == "quantinuum" and hqc_values:
        stats_rows.append(("HQC per circuit", *_stat_tuple(hqc_values)))

    print()
    print(
        tabulate(
            stats_rows,
            headers=["Per-circuit metric", "Min", "Max", "Avg"],
            tablefmt="github",
        )
    )
