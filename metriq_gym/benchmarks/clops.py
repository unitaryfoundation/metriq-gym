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
    - [Wack et al., "Quality, Speed, and Scale: three key attributes to measure the performance of near-term quantum computers", arXiv:2110.14108](https://arxiv.org/abs/2110.14108).
    - [Qiskit Device Benchmarking CLOPS](https://github.com/qiskit-community/qiskit-device-benchmarking).
"""

from collections import deque
from dataclasses import dataclass

import rustworkx as rx
import numpy as np
from typing import TYPE_CHECKING, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RZGate, SXGate

from pydantic import Field
from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.qplatform.job import execution_time
from metriq_gym.qplatform.device import (
    connectivity_graph,
    connectivity_graph_for_gate,
    pruned_connectivity_graph,
)
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


def create_qubit_list(width: int, topology_graph: rx.PyGraph) -> list[int]:
    """
    Returns a list of 'width' connected qubits from the topology_graph.

    Args:
        width: Number of connected qubits to find.
        topology_graph: The device topology as a rustworkx PyGraph.

    Note: Adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::create_qubit_map
        But assumes the faulty qubits/edges were already pruned from the graph.
    """
    total_qubits = topology_graph.num_nodes()
    if total_qubits < width:
        raise ValueError(f"Device has only {total_qubits} qubits, cannot create set of {width}")
    for starting_qubit in topology_graph.node_indices():
        visited: set[int] = set()
        queue = deque([starting_qubit])
        while queue and len(visited) < width:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in topology_graph.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(visited) >= width:
            return list(visited)[:width]
    raise ValueError(f"Insufficient connected qubits to create set of {width} qubits")


def append_1q_layer(
    circuit, qubits: List[int], parameterized: bool = True, parameter_prefix="θ"
) -> List[ParameterVector]:
    """Append a layer of parameterized 1-qubit gates on specified qubits.

    Args:
        circuit: The QuantumCircuit to append gates to.
        qubits: List of qubit indices to apply the gates on.
        parameterized: If True, append parameterized gates; if False, append fixed gates.
        parameter_prefix: Prefix for parameter names if parameterized is True.

    Note: Adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::append_1q_layer
    and submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::_append_1q_layer_rzsx

    The current version of the qiskit-device-benchmarking code uses the default RZSX basis for these gates, so for
    easy of understanding the code, we explicitly inline that function here.
    """
    size = len(qubits)
    pars0 = ParameterVector(f"{parameter_prefix}_0", size)
    pars1 = ParameterVector(f"{parameter_prefix}_1", size)
    pars2 = ParameterVector(f"{parameter_prefix}_2", size)

    for i, q in enumerate(qubits):
        if parameterized:
            circuit._append(RZGate(pars0[i]), [q], [])
            circuit._append(SXGate(), [q], [])
            circuit._append(RZGate(pars1[i]), [q], [])
            circuit._append(SXGate(), [q], [])
            circuit._append(RZGate(pars2[i]), [q], [])
        else:
            circuit._append(SXGate(), [q], [])

    return [pars0, pars1, pars2]


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
    seed: int = 0,
) -> list[QuantumCircuit]:
    qubit_map = create_qubit_list(width, topology_graph)

    # remove edges beyond the width of the circuit we are trying to generate
    for edge in topology_graph.edge_list():
        if edge[0] not in qubit_map or edge[1] not in qubit_map:
            topology_graph.remove_edge(*edge)

    qc = QuantumCircuit(max(qubit_map) + 1, max(qubit_map) + 1)
    qubits = [qc.qubits[i] for i in qubit_map]

    parameters = []
    rng = np.random.default_rng(seed)
    for d in range(layers):
        append_2q_layer(qc, topology_graph, two_qubit_gate, rng)

        # add barrier to form "twirling box" to inform primitve where layers are for twirled gates
        qc.barrier(qubits)

        parameters += append_1q_layer(qc, qubits, parameterized=True, parameter_prefix=f"L{d}")

    qc.barrier(qubits)
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
        # If the device has restricted connectivity for the two-qubit gate, use
        # that restricted topology to create the chain
        graph = connectivity_graph_for_gate(device, self.params.two_qubit_gate)
        if graph is None:
            graph = connectivity_graph(device)

        graph = pruned_connectivity_graph(device, graph)

        circuits = prepare_clops_circuits(
            width=self.params.num_qubits,
            layers=self.params.num_layers,
            num_circuits=self.params.num_circuits,
            two_qubit_gate=self.params.two_qubit_gate,
            seed=self.params.seed,
            topology_graph=graph,
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
