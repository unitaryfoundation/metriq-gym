"""Utilities for estimating resource requirements of metriq-gym benchmarks."""

from dataclasses import dataclass, field
from typing import Callable, Iterable

import networkx as nx
import rustworkx as rx
from pydantic import BaseModel

from qbraid import QuantumDevice
from qiskit import QuantumCircuit
from tabulate import tabulate

from metriq_gym.benchmarks.bseq import generate_chsh_circuit_sets
from metriq_gym.benchmarks.clops import prepare_clops_circuits
from metriq_gym.benchmarks.lr_qaoa import prepare_qaoa_circuit
from metriq_gym.benchmarks.mirror_circuits import (
    create_subgraph_from_qubits,
    generate_mirror_circuit,
    select_optimal_qubit_subset,
)
from metriq_gym.benchmarks.qedc_benchmarks import get_circuits_and_metrics
from metriq_gym.benchmarks.qml_kernel import create_inner_product_circuit
from metriq_gym.benchmarks.quantum_volume import prepare_qv_circuits
from metriq_gym.benchmarks.wit import wit_circuit
from metriq_gym.circuits import EncodingType, GraphType
from metriq_gym.constants import JobType
from metriq_gym.helpers.graph_helpers import device_graph_coloring
from metriq_gym.qplatform.device import connectivity_graph


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


def _aggregate(
    batches: Iterable[CircuitBatch],
    hqc_fn: HQCFunction | None,
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


def _get_shots(params: BaseModel) -> int:
    for f in ("shots", "num_shots"):
        value = getattr(params, f, None)
        if value is not None:
            return int(value)
    raise ValueError("Parameters do not define a shots/num_shots field.")


def _estimate_wit(params: BaseModel, _device: QuantumDevice | None) -> list[CircuitBatch]:
    circuit = wit_circuit(params.num_qubits)
    return [CircuitBatch(circuits=[circuit], shots=_get_shots(params))]


def _estimate_qml_kernel(params: BaseModel, _device: QuantumDevice | None) -> list[CircuitBatch]:
    circuit = create_inner_product_circuit(params.num_qubits, seed=getattr(params, "seed", 0))
    return [CircuitBatch(circuits=[circuit], shots=_get_shots(params))]


def _estimate_quantum_volume(
    params: BaseModel, _device: QuantumDevice | None
) -> list[CircuitBatch]:
    circuits, _ = prepare_qv_circuits(params.num_qubits, params.trials)
    return [CircuitBatch(circuits=circuits, shots=_get_shots(params))]


def _estimate_bseq(params: BaseModel, device: QuantumDevice | None) -> list[CircuitBatch]:
    device = _require_device(device, "BSEQ")
    topology_graph = connectivity_graph(device)
    coloring = device_graph_coloring(topology_graph)
    circuit_sets = generate_chsh_circuit_sets(coloring)
    return [
        CircuitBatch(circuits=circuit_group, shots=params.shots) for circuit_group in circuit_sets
    ]


def _estimate_clops(params: BaseModel, device: QuantumDevice | None) -> list[CircuitBatch]:
    device = _require_device(device, "CLOPS")
    topology_graph = connectivity_graph(device)
    num_qubits = device.num_qubits
    if num_qubits is None:
        raise ValueError("Device must have known qubit count for CLOPS resource estimation.")
    basis_gates = set(device.profile.basis_gates or [])
    circuits = prepare_clops_circuits(
        width=params.width,
        layers=params.num_layers,
        num_circuits=params.num_circuits,
        basis_gates=basis_gates,
        topology_graph=topology_graph,
        total_qubits=num_qubits,
    )
    return [CircuitBatch(circuits=circuits, shots=params.shots)]


def _estimate_mirror_circuits(
    params: BaseModel, device: QuantumDevice | None
) -> list[CircuitBatch]:
    device = _require_device(device, "Mirror Circuits")
    topology_graph = connectivity_graph(device)
    target_width = getattr(params, "width", None)
    if not isinstance(target_width, (int, type(None))):
        target_width = None

    if target_width is not None:
        max_width = len(topology_graph.node_indices())
        if target_width > max_width:
            raise ValueError(f"Requested width {target_width} exceeds device capacity {max_width}")
        selected_qubits = select_optimal_qubit_subset(topology_graph, target_width)
        working_graph = create_subgraph_from_qubits(topology_graph, selected_qubits)
    else:
        working_graph = topology_graph

    circuits: list[QuantumCircuit] = []
    num_circuits = params.num_circuits
    for idx in range(num_circuits):
        circuit_seed = None if params.seed is None else params.seed + idx
        circuit, _ = generate_mirror_circuit(
            num_layers=params.num_layers,
            two_qubit_gate_prob=params.two_qubit_gate_prob,
            connectivity_graph=working_graph,
            two_qubit_gate_name=params.two_qubit_gate_name,
            seed=circuit_seed,
        )
        circuits.append(circuit)

    return [CircuitBatch(circuits=circuits, shots=params.shots)]


def _estimate_qedc(params: BaseModel, _device: QuantumDevice | None) -> list[CircuitBatch]:
    circuits, _, _ = get_circuits_and_metrics(
        benchmark_name=params.benchmark_name,
        params=params.model_dump(exclude={"benchmark_name"}),
    )
    flat_circuits: list[QuantumCircuit] = []
    for num_qubits in circuits:
        for circuit in circuits[num_qubits].values():
            flat_circuits.append(circuit)
    return [CircuitBatch(circuits=flat_circuits, shots=_get_shots(params))]


def _estimate_lr_qaoa(params: BaseModel, device: QuantumDevice | None) -> list[CircuitBatch]:
    import random

    random.seed(params.seed)

    num_qubits = params.num_qubits
    graph_type: GraphType = params.graph_type
    qaoa_layers = params.qaoa_layers
    circuit_encoding: EncodingType = "Direct"

    if device is None:
        raise ValueError("LR-QAOA benchmark requires a device to estimate resources.")

    if device.id == "aer_simulator" and graph_type == "NL":
        graph_device = rx.generators.star_graph(num_qubits)
    else:
        graph_device = connectivity_graph(device)

    edges_device = list(graph_device.edge_list())

    if graph_type == "1D":
        edges = [(i, i + 1) for i in range(num_qubits - 1)]
    elif graph_type == "NL":
        if graph_device.num_nodes() != num_qubits:
            raise ValueError("Number of qubits does not match the device connectivity graph.")
        edges = [(edge[0], edge[1]) for edge in edges_device]
    elif graph_type == "FC":
        edges = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        device_edges = {(edge[0], edge[1]) for edge in edges_device}
        if not device_edges.issuperset(edges):
            circuit_encoding = "SWAP"
    else:
        raise ValueError(
            f"Unsupported graph type: {graph_type}. Supported types are '1D', 'NL', and 'FC'."
        )

    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    graph_info = [(i, j, random.choice(possible_weights)) for i, j in edges]
    graph.add_weighted_edges_from(graph_info)

    circuits = prepare_qaoa_circuit(
        graph=graph,
        qaoa_layers=qaoa_layers,
        graph_type=graph_type,
        circuit_encoding=circuit_encoding,
    )

    circuits_with_params: list[QuantumCircuit] = []
    for _ in range(params.trials):
        for p_layer, circuit in zip(qaoa_layers, circuits):
            linear_ramp = list(range(1, p_layer + 1))
            betas = [i * params.delta_beta / p_layer for i in reversed(linear_ramp)]
            gammas = [i * params.delta_gamma / p_layer for i in linear_ramp]
            circuits_with_params.append(circuit.assign_parameters(betas + gammas))

    return [CircuitBatch(circuits=circuits_with_params, shots=params.shots)]


EstimatorFunction = Callable[[BaseModel, QuantumDevice | None], list[CircuitBatch]]


ESTIMATORS: dict[JobType, EstimatorFunction] = {
    JobType.WIT: _estimate_wit,
    JobType.QML_KERNEL: _estimate_qml_kernel,
    JobType.QUANTUM_VOLUME: _estimate_quantum_volume,
    JobType.BSEQ: _estimate_bseq,
    JobType.CLOPS: _estimate_clops,
    JobType.MIRROR_CIRCUITS: _estimate_mirror_circuits,
    JobType.BERNSTEIN_VAZIRANI: _estimate_qedc,
    JobType.PHASE_ESTIMATION: _estimate_qedc,
    JobType.HIDDEN_SHIFT: _estimate_qedc,
    JobType.QUANTUM_FOURIER_TRANSFORM: _estimate_qedc,
    JobType.LR_QAOA: _estimate_lr_qaoa,
}


def estimate_resources(
    job_type: JobType,
    params: BaseModel,
    device: QuantumDevice | None,
    hqc_fn: HQCFunction | None = None,
) -> ResourceEstimate:
    if job_type not in ESTIMATORS:
        raise NotImplementedError(f"Resource estimation not implemented for {job_type.value}")

    batches = ESTIMATORS[job_type](params, device)
    return _aggregate(batches, hqc_fn)


def quantinuum_hqc_formula(counts: GateCounts, shots: int) -> float:
    """Compute Quantinuum HQCs using the published formula."""

    n_one = counts.one_qubit
    n_two = counts.two_qubit
    # Includes the implicit initial state preparation (+1) and any resets.
    n_measure = counts.measurements + counts.resets + 1

    gate_term = n_one + 10 * n_two + 5 * n_measure
    return 5.0 + shots * gate_term / 5000.0


def _stat_tuple(values: list[int | float]) -> tuple[str, str, str]:
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
