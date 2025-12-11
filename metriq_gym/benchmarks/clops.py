"""CLOPS (circuit layer operations per second) benchmark implementation."""

import copy
from dataclasses import dataclass

import rustworkx as rx
import numpy as np
from typing import TYPE_CHECKING
from qiskit import QuantumCircuit
from qiskit_device_benchmarking.clops.clops_benchmark import append_1q_layer

from pydantic import Field
from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.qplatform.job import execution_time
from metriq_gym.qplatform.device import connectivity_graph
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


@dataclass
class ClopsData(BenchmarkData):
    pass


class ClopsResult(BenchmarkResult):
    clops_score: float = Field(...)

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=self.clops_score)


# adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::create_qubit_map
# As opposed to the original version of this function, this takes a topology graph as an argument instead of an IBM's coupling map object.
def create_qubit_map(width: int, topology_graph: rx.PyGraph, total_qubits: int) -> list[int]:
    """
    Returns a list of  'width' qubits that are connected based on a coupling map that has
    bad edges already removed and a list of faulty qubits.  If there is not
    such a map, raises ValueError
    """
    qubit_map = []

    # make coupling map bidirectional so we can find all neighbors
    cm = copy.deepcopy(topology_graph)
    for edge in topology_graph.edge_list():
        cm.add_edge(edge[1], edge[0], None)

    for starting_qubit in range(total_qubits):
        qubit_map = [starting_qubit]
        new_neighbors = []
        prospective_neighbors = list(cm.neighbors(starting_qubit))
        while prospective_neighbors:
            for pn in prospective_neighbors:
                if pn not in qubit_map:
                    new_neighbors.append(pn)
            qubit_map = qubit_map + new_neighbors
            prospective_neighbors = []
            for nn in new_neighbors:
                potential_neighbors = list(cm.neighbors(nn))
                for pn in potential_neighbors:
                    if pn not in prospective_neighbors:
                        prospective_neighbors.append(pn)
            new_neighbors = []
            if len(qubit_map) >= width:
                return qubit_map[:width]
    raise ValueError(f"Insufficient connected qubits to create set of {width} qubits")


# adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::prepare_clops_circuits
# As opposed to the original version of this function, this takes a topology graph as an argument instead of an IBM's coupling map object.
def append_2q_layer(
    qc: QuantumCircuit, topology_graph: rx.PyGraph, basis_gates: set[str], rng: np.random.Generator
) -> None:
    """
    Add a layer of random 2q gates.
    """
    available_edges = set(topology_graph.edge_list())
    while len(available_edges) > 0:
        edge = tuple(rng.choice(list(available_edges)))
        available_edges.remove(edge)
        edges_to_delete = set()
        for ce in list(available_edges):
            if (edge[0] in ce) or (edge[1] in ce):
                edges_to_delete.add(ce)
        available_edges.difference_update(edges_to_delete)
        if "ecr" in basis_gates:
            qc.ecr(*edge)
        elif "cz" in basis_gates:
            qc.cz(*edge)
        else:
            qc.cx(*edge)


# adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::prepare_clops_circuits
def prepare_clops_circuits(
    width: int,
    layers: int,
    num_circuits: int,
    basis_gates: set[str],
    topology_graph: rx.PyGraph,
    total_qubits: int,
    seed: int = 0,
) -> list[QuantumCircuit]:
    qubit_map = create_qubit_map(width, topology_graph, total_qubits=total_qubits)

    for edge in topology_graph.edge_list():
        if edge[0] not in qubit_map or edge[1] not in qubit_map:
            topology_graph.remove_edge(*edge)

    qc = QuantumCircuit(max(qubit_map) + 1, max(qubit_map) + 1)
    qubits = [qc.qubits[i] for i in qubit_map]

    parameters = []
    rng = np.random.default_rng(seed)
    for d in range(layers):
        append_2q_layer(qc, topology_graph, basis_gates, rng)
        parameters += append_1q_layer(qc, qubits, parameterized=True, parameter_prefix=f"L{d}")

    for idx in range(width):
        qc.measure(qubit_map[idx], idx)

    # Parameters are instantiated for each circuit with random values
    parametrized_circuits = [
        qc.assign_parameters(
            [rng.uniform(0, np.pi * 2) for _ in range(sum([len(param) for param in parameters]))]
        )
        for _ in range(num_circuits)
    ]

    return parametrized_circuits


class Clops(Benchmark):
    """
    Circuit Layer Operations per Second Benchmark
    https://arxiv.org/abs/2110.14108
    """

    def _build_circuits(self, device: "QuantumDevice") -> list[QuantumCircuit]:
        """Shared circuit construction logic.

        Args:
            device: The quantum device to build circuits for.

        Returns:
            List of CLOPS circuits.
        """
        topology_graph = connectivity_graph(device)
        num_qubits = device.num_qubits
        if num_qubits is None:
            raise ValueError(
                "Device must have a known number of qubits to run the CLOPS benchmark."
            )
        basis_gates = set(device.profile.basis_gates or [])
        circuits = prepare_clops_circuits(
            width=self.params.width,
            layers=self.params.num_layers,
            num_circuits=self.params.num_circuits,
            basis_gates=basis_gates,
            topology_graph=topology_graph,
            total_qubits=num_qubits,
        )
        return circuits

    def dispatch_handler(self, device: "QuantumDevice") -> ClopsData:
        circuits = self._build_circuits(device)
        return ClopsData.from_quantum_job(device.run(circuits, shots=self.params.shots))

    def poll_handler(
        self,
        job_data: ClopsData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> ClopsResult:
        clops_score = (self.params.num_circuits * self.params.num_layers * self.params.shots) / sum(
            execution_time(quantum_job) for quantum_job in quantum_jobs
        )
        return ClopsResult(clops_score=clops_score)

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list["CircuitBatch"]:
        circuits = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
