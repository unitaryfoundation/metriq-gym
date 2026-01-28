"""CLOPS (Circuit Layer Operations Per Second) benchmark implementation.

Summary:
    Measures the throughput of a quantum system by timing the execution of parameterized
    quantum volume-style circuits. CLOPS captures end-to-end performance including
    compilation, communication, and execution overhead.

Result interpretation:
    Polling returns ClopsResult with:
        - clops_score: circuit layer operations per second (higher is better).
    This metric reflects real-world workload performance rather than isolated gate speeds.

References:
    - [Wack et al., "Quality, Speed, and Scale: three key attributes to measure the
      performance of near-term quantum computers", arXiv:2110.14108](https://arxiv.org/abs/2110.14108).
    - [Qiskit Device Benchmarking CLOPS](https://github.com/qiskit-community/qiskit-device-benchmarking).
"""

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
from metriq_gym.qplatform.device import connectivity_graph, connectivity_graph_for_gate
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


def create_qubit_map(width: int, topology_graph: rx.PyGraph, total_qubits: int) -> list[int]:
    """
    Returns a list of 'width' connected qubits from the topology_graph.

    Args:
        width: Number of connected qubits to find.
        topology_graph: The device topology as a rustworkx PyGraph.
        total_qubits: Total number of qubits in the device.

    Note: Adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::create_qubit_map
    Unlike that version, this does not attempt to filter out faulty qubits, which we do not yet
    have a cross-platform approach for.
    """
    for starting_qubit in range(total_qubits):
        visited: set[int] = set()
        queue = [starting_qubit]
        while queue and len(visited) < width:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in topology_graph.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(visited) >= width:
            return list(visited)[:width]
    raise ValueError(f"Insufficient connected qubits to create set of {width} qubits")


def append_2q_layer(
    qc: QuantumCircuit, topology_graph: rx.PyGraph, two_qubit_gate: str, rng: np.random.Generator
) -> None:
    """
    Add a layer of random 2q gates.

    Args:
        qc: The quantum circuit to append to.
        topology_graph: The device topology as a rustworkx PyGraph.
        two_qubit_gate: The type of 2q gate to use ('ecr', 'cz', 'cx').
        rng: Random number generator

    Note: Adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::append_2q_layer
    This version takes in the cross-platform PyGraph instead of a Qiskit CouplingMap, and also
    takes an explicit two_qubit_gate versus an available set.
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
        if "ecr" == two_qubit_gate:
            qc.ecr(*edge)
        elif "cz" == two_qubit_gate:
            qc.cz(*edge)
        elif "cx" == two_qubit_gate:
            qc.cx(*edge)
        else:
            raise ValueError(f"Unsupported two qubit gate: {two_qubit_gate}")


# adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::prepare_clops_circuits
def prepare_clops_circuits(
    width: int,
    layers: int,
    num_circuits: int,
    two_qubit_gate: str,
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
        append_2q_layer(qc, topology_graph, two_qubit_gate, rng)
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
        num_qubits = device.num_qubits
        if num_qubits is None:
            raise ValueError(
                "Device must have a known number of qubits to run the CLOPS benchmark."
            )

        # If the device has restricted connectivity for the 2 gate, use
        # that restricted topology to create the chain
        graph = connectivity_graph_for_gate(device, self.params.two_qubit_gate)
        if graph is None:
            graph = connectivity_graph(device)

        circuits = prepare_clops_circuits(
            width=self.params.num_qubits,
            layers=self.params.num_layers,
            num_circuits=self.params.num_circuits,
            two_qubit_gate=self.params.two_qubit_gate,
            seed=self.params.seed,
            topology_graph=graph,
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
