"""CLOPS (Circuit Layer Operations Per Second) benchmark implementation.

Summary:
    Measures the throughput of a quantum system by timing the execution of parameterized
    quantum volume-style circuits. CLOPS captures end-to-end performance including
    compilation, communication, and execution overhead.


Result interpretation:
    Polling returns ClopsResult with:
         - clops_score: circuit layer operations per second (higher is better) as measured from
           time of submission to job completion as reported by the cloud platform.
        - steady_state_clops: circuit layer operations per second (higher is better), ignorning the
            first execution span to exclude pipeline startup costs, measuring sustained throughput.
            Only works for cloud platforms that provide execution span metadata (currently only IBM Runtime), and is None when spans are unavailable or there are fewer than two spans.

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

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
import logging

from metriq_gym.qplatform.job import execution_time
from metriq_gym.qplatform.device import (
    connectivity_graph,
    connectivity_graph_for_gate,
    pruned_connectivity_graph,
)
from metriq_gym.resource_estimation import CircuitBatch
from metriq_gym.ibm_sampler.device import IBMSamplerDevice

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob

logger = logging.getLogger(__name__)


@dataclass
class ClopsData(BenchmarkData):
    pass


class ClopsResult(BenchmarkResult):
    clops_score: float
    steady_state_clops: float | None = None

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
) -> List[ParameterVector] | None:
    """Append a layer of parameterized 1-qubit gates on specified qubits.

    Args:
        circuit: The QuantumCircuit to append gates to.
        qubits: List of qubit indices to apply the gates on.
        parameterized: If True, append parameterized gates; if False, append fixed gates.
        parameter_prefix: Prefix for parameter names if parameterized is True.

    Returns:
        If parameterized is True, returns a list of ParameterVector objects for the parameters used.
        If parameterized is False, returns None.

    Note: Adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::append_1q_layer
    and submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::_append_1q_layer_rzsx

    The current version of the qiskit-device-benchmarking code uses the default RZSX basis for these gates, so for
    ease of understanding the code, we explicitly inline that function here.
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

    if parameterized:
        return [pars0, pars1, pars2]
    return None


def append_2q_layer(
    qc: QuantumCircuit, topology_graph: rx.PyGraph, two_qubit_gate: str, rng: np.random.Generator
) -> None:
    """
    Add a layer of random 2q gates, where

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


def prepare_clops_template(
    width: int,
    layers: int,
    two_qubit_gate: str,
    topology_graph: rx.PyGraph,
    parameterized: bool = True,
    seed: int = 0,
) -> tuple[QuantumCircuit, list[ParameterVector]]:
    """Build a single CLOPS template circuit.

    Args:
        width: Number of qubits.
        layers: Number of 2Q+1Q layer repetitions.
        two_qubit_gate: Native two-qubit gate name.
        topology_graph: Device connectivity (will be pruned in-place).
        parameterized: If True, 1Q gates are parameterized; if False, fixed SX
            gates are used (appropriate for twirled mode where the Sampler
            randomizes the gates).
        seed: RNG seed for reproducible 2Q gate placement.

    Returns:
        A tuple of (template_circuit, parameters) where *parameters* is a
        flat list of ``ParameterVector`` objects (empty when
        ``parameterized=False``).
    Note: Adapted from submodules/qiskit-device-benchmarking/qiskit_device_benchmarking/clops/clops_benchmark.py::prepare_clops_circuits
    """
    qubit_map = create_qubit_list(width, topology_graph)

    # remove edges beyond the width of the circuit we are trying to generate
    for edge in topology_graph.edge_list():
        if edge[0] not in qubit_map or edge[1] not in qubit_map:
            topology_graph.remove_edge(*edge)

    qc = QuantumCircuit(max(qubit_map) + 1, max(qubit_map) + 1)
    qubits = [qc.qubits[i] for i in qubit_map]

    parameters: list[ParameterVector] = []
    rng = np.random.default_rng(seed)
    for d in range(layers):
        append_2q_layer(qc, topology_graph, two_qubit_gate, rng)

        # add barrier to form "twirling box" to inform primitive where layers are for twirled gates
        qc.barrier(qubits)

        params = append_1q_layer(qc, qubits, parameterized=parameterized, parameter_prefix=f"L{d}")
        if params is not None:
            parameters += params

    qc.barrier(qubits)
    for idx in range(width):
        qc.measure(qubit_map[idx], idx)

    return qc, parameters


def instantiate_circuits(
    template: QuantumCircuit,
    parameters: list[ParameterVector],
    num_circuits: int,
    seed: int = 0,
) -> list[QuantumCircuit]:
    """Bind random parameter values to produce *num_circuits* concrete circuits."""
    rng = np.random.default_rng(seed)
    num_params = sum(len(p) for p in parameters)
    return [
        template.assign_parameters([rng.uniform(0, np.pi * 2) for _ in range(num_params)])
        for _ in range(num_circuits)
    ]


def _compute_steady_state_clops(quantum_jobs: list["QuantumJob"], num_layers: int) -> float | None:
    """Compute steady-state CLOPS from IBM Runtime ExecutionSpan metadata.

    Skips the first execution span to exclude pipeline startup costs,
    measuring sustained throughput from the end of the first sub-job
    to the end of the last sub-job.

    Adapted from qiskit-device-benchmarking ``_clops_throughput_sampler``.

    Returns:
        Steady-state CLOPS value, or *None* if execution span data is
        unavailable or there are fewer than two spans.
    """
    from qbraid.runtime import QiskitJob
    from qiskit_ibm_runtime.execution_span import ExecutionSpans

    for quantum_job in quantum_jobs:
        if not isinstance(quantum_job, QiskitJob):
            continue

        try:
            result = quantum_job._job.result()
            execution_spans: ExecutionSpans = result.metadata["execution"]["execution_spans"]
        except (AttributeError, KeyError, TypeError):
            logger.debug("No execution spans available for job %s", quantum_job.id)
            continue

        spans = execution_spans.sort()
        if len(spans) < 2:
            logger.debug(
                "Only %d execution span(s); need ≥2 for steady-state CLOPS.",
                len(spans),
            )
            return None

        total_size = sum(span.size for span in spans)
        first_span_stop = spans[0].stop
        last_span_stop = spans.stop

        elapsed = (last_span_stop - first_span_stop).total_seconds()
        if elapsed <= 0:
            return None

        return round(((total_size - spans[0].size) * num_layers) / elapsed)

    return None


class Clops(Benchmark):
    """
    Circuit Layer Operations per Second Benchmark
    https://arxiv.org/abs/2110.14108
    """

    def _get_topology(self, device: "QuantumDevice") -> rx.PyGraph:
        """Return the (possibly pruned) device topology for the configured 2Q gate."""
        graph = connectivity_graph_for_gate(device, self.params.two_qubit_gate)
        if graph is None:
            graph = connectivity_graph(device)
        return pruned_connectivity_graph(device, graph)

    def _build_template(
        self, device: "QuantumDevice", parameterized: bool = True
    ) -> tuple[QuantumCircuit, list[ParameterVector]]:
        """Build the CLOPS template circuit from device topology."""
        graph = self._get_topology(device)
        return prepare_clops_template(
            width=self.params.num_qubits,
            layers=self.params.num_layers,
            two_qubit_gate=self.params.two_qubit_gate,
            topology_graph=graph,
            parameterized=parameterized,
            seed=self.params.seed,
        )

    # ------------------------------------------------------------------
    # Mode-specific dispatch helpers
    # ------------------------------------------------------------------

    def _dispatch_instantiated(self, device: "QuantumDevice") -> ClopsData:
        """Bind parameters locally and send concrete circuits (any provider)."""
        template, parameters = self._build_template(device, parameterized=True)
        circuits = instantiate_circuits(
            template, parameters, self.params.num_circuits, seed=self.params.seed
        )
        if isinstance(device, IBMSamplerDevice):
            return ClopsData.from_quantum_job(
                device.run(circuits, shots=self.params.shots, use_session=self.params.use_session)
            )
        return ClopsData.from_quantum_job(device.run(circuits, shots=self.params.shots))

    def _dispatch_parameterized(self, device: "IBMSamplerDevice") -> ClopsData:
        """Send a single parameterized circuit with parameter arrays (ibm_sampler).

        Follows the SamplerV2 PUB format: ``(circuit, parameter_values, shots)``.
        See: https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2
        """
        template, parameters = self._build_template(device, parameterized=True)

        num_params = sum(len(p) for p in parameters)
        rng = np.random.default_rng(self.params.seed)
        param_values = [
            [rng.uniform(0, np.pi * 2) for _ in range(num_params)]
            for _ in range(self.params.num_circuits)
        ]

        pub = (template, param_values, self.params.shots)
        return ClopsData.from_quantum_job(
            device.submit(pubs=[pub], use_session=self.params.use_session)
        )

    def _dispatch_twirled(self, device: "IBMSamplerDevice") -> ClopsData:
        """Send a fixed circuit and delegate randomization to the Sampler twirler (ibm_sampler).

        The Sampler applies gate twirling to produce ``num_circuits``
        randomized variants, each run for ``shots`` shots.
        """
        from qiskit_ibm_runtime.options import TwirlingOptions

        template, _parameters = self._build_template(device, parameterized=False)

        twirling_opts = TwirlingOptions(
            num_randomizations=self.params.num_circuits,
            shots_per_randomization=self.params.shots,
            enable_gates=True,
        )
        return ClopsData.from_quantum_job(
            device.submit(
                pubs=[template],
                shots=self.params.shots * self.params.num_circuits,
                twirling_options=twirling_opts,
                use_session=self.params.use_session,
            )
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def dispatch_handler(self, device: "QuantumDevice") -> ClopsData:
        mode = getattr(self.params, "mode", "instantiated") or "instantiated"

        if mode in ("parameterized", "twirled"):
            if not isinstance(device, IBMSamplerDevice):
                raise ValueError(
                    f"CLOPS mode '{mode}' requires the ibm_sampler provider "
                    f"(IBMSamplerDevice), but got {type(device).__name__}."
                )
        if self.params.use_session and not isinstance(device, IBMSamplerDevice):
            raise ValueError(
                f"CLOPS parameter 'use_session=True' requires the ibm_sampler provider "
                f"(IBMSamplerDevice), but got {type(device).__name__}."
            )

        if mode == "instantiated":
            return self._dispatch_instantiated(device)
        elif mode == "parameterized":
            return self._dispatch_parameterized(device)
        elif mode == "twirled":
            return self._dispatch_twirled(device)
        else:
            raise ValueError(f"Unknown CLOPS mode: '{mode}'")

    def poll_handler(
        self,
        job_data: ClopsData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> ClopsResult:
        clops_score = (self.params.num_circuits * self.params.num_layers * self.params.shots) / sum(
            execution_time(quantum_job) for quantum_job in quantum_jobs
        )
        steady_state = _compute_steady_state_clops(quantum_jobs, self.params.num_layers)
        return ClopsResult(clops_score=clops_score, steady_state_clops=steady_state)

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list["CircuitBatch"]:
        """
        Estimates resources needed for the instantiated mode only, as the
        parameterized and twirled modes require features specific to the
        cloud platform that are not yet supported.
        """
        if self.params.mode != "instantiated":
            raise NotImplementedError(
                f"Resource estimation for mode '{self.params.mode}' is not implemented."
            )

        template, parameters = self._build_template(device, parameterized=True)
        circuits = instantiate_circuits(
            template, parameters, self.params.num_circuits, seed=self.params.seed
        )
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
