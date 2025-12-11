import math
import statistics
import networkx as nx
import numpy as np
import rustworkx as rx
from scipy import stats
from dataclasses import dataclass
import random
import dimod
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler
from typing import TYPE_CHECKING
from qiskit import QuantumCircuit

from metriq_gym.circuits import qaoa_circuit
from metriq_gym.circuits import GraphType, EncodingType

from pydantic import Field
from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.qplatform.device import connectivity_graph
from metriq_gym.resource_estimation import CircuitBatch

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob
    from qbraid.runtime.result_data import MeasCount


def weighted_maxcut_solver(graph: nx.Graph) -> str:
    """Constructs a mathematical model for the Max-Cut problem using the Simulated Annealing solver.
    This function creates a binary optimization model to maximize the cut in a weighted graph.
    Args:
        graph (networkx.Graph): The input weighted graph where edges represent cut costs.
    Returns:
        str: A binary string representing the optimal partition of the graph nodes (e.g., "1010").
    """
    num_nodes = graph.number_of_nodes()
    bqp = dimod.BQM.from_ising({}, {(i, j): weight for i, j, weight in graph.edges(data="weight")})
    result = SimulatedAnnealingSampler().sample(bqp)
    max_cost = -1.0
    for sample in result:
        bitstring = "".join(("0" if sample[i] == -1 else "1") for i in range(num_nodes))
        cost = cost_maxcut(bitstring, graph)
        if cost > max_cost:
            max_cost = cost
            optimal_solution = bitstring
    return optimal_solution


def cost_maxcut(bitstring: str, graph: nx.Graph) -> float:
    """
    Computes the cost of a given bitstring solution for the Max-Cut problem.

    Parameters:
    bitstring (str): A binary string representing a partition of the graph nodes (e.g., "1010").
    graph (networkx.Graph): The input weighted graph where edges represent cut costs.

    Returns:
    float: The computed cost of the Max-Cut solution.
    """
    if len(bitstring) != graph.number_of_nodes():
        raise ValueError(
            f"Bitstring length: {len(bitstring)} must match the number of nodes in the graph: {graph.number_of_nodes()}."
        )
    cost = 0
    for i, j in graph.edges():
        if bitstring[i] + bitstring[j] in ["10", "01"]:
            cost += graph[i][j]["weight"]
    return cost


def objective_func(samples_dict: dict, graph: nx.Graph, optimal: str) -> dict:
    """
    Evaluates the performance of LR-QAOA for the Max-Cut problem.

    Parameters:
    samples_dict (dict): A dictionary where keys are bitstrings (binary solutions),
                         and values are their occurrence counts.
    graph (networkx.Graph): The input weighted graph where edges represent cut costs.
    optimal (str): The optimal bitstring solution found by classical solvers (e.g., CPLEX).

    Returns:
    dict: A dictionary containing:
        - "approx_ratio": The approximation ratio.
        - "optimal_probability": The optimal_probability of sampling the optimal solution.
    """

    max_cost = cost_maxcut(optimal, graph)
    optimal_probability = 0.0
    total_cost = 0.0
    shots = 0.0
    for bitstring, counts in samples_dict.items():
        cost = cost_maxcut(bitstring, graph)
        total_cost += counts * cost
        if math.isclose(cost, max_cost):
            optimal_probability += counts

        if cost > max_cost:
            print(f"There is a better cost than that of Simulated Annealing: {cost - max_cost}")
        shots += counts
    approx_ratio = total_cost / (max_cost * shots)
    optimal_probability /= shots
    return {"approx_ratio": approx_ratio, "optimal_probability": optimal_probability}


def random_samples(num_samples: int, n_qubits: int) -> dict:
    """
    Generates random bitstring samples for a given number of qubits.

    Parameters:
    num_samples (int): The number of random bitstrings to generate.
    n_qubits (int): The number of qubits (length of each bitstring).

    Returns:
    dict: A dictionary where keys are randomly generated bitstrings
          and values are their occurrence counts.
    """

    random_samples = {}

    for _ in range(num_samples):
        bitstring = "".join(random.choice(["0", "1"]) for _ in range(n_qubits))
        if bitstring not in random_samples:
            random_samples[bitstring] = 0
        random_samples[bitstring] += 1

    return random_samples


def calc_random_stats(num_qubits, graph_info, shots, num_random_trials, optimal_sol):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    graph.add_weighted_edges_from(graph_info)
    approx_ratio_random_list = []
    for _ in range(num_random_trials):
        random_samples_dict = random_samples(shots, len(graph.nodes()))
        approx_ratio_random_list.append(
            objective_func(random_samples_dict, graph, optimal_sol)["approx_ratio"]
        )
    approx_ratio_random_mean = statistics.mean(approx_ratio_random_list)
    approx_ratio_random_std = statistics.pstdev(approx_ratio_random_list)
    return approx_ratio_random_mean, approx_ratio_random_std


@dataclass
class LinearRampQAOAData(BenchmarkData):
    provider_job_ids: list[str]
    num_qubits: int
    graph_info: list[list]
    graph_type: GraphType
    optimal_sol: str
    trials: int
    num_random_trials: int
    confidence_level: float
    shots: int
    qaoa_layers: list[int]
    seed: int
    delta_beta: float
    delta_gamma: float
    approx_ratio_random_mean: float
    approx_ratio_random_std: float
    circuit_encoding: EncodingType


class LinearRampQAOAResult(BenchmarkResult):
    approx_ratio: list[float]
    random_approx_ratio: float = Field(...)
    confidence_pass: list[bool]
    effective_approx_ratio: list[float] | None = None

    def compute_score(self) -> BenchmarkScore:
        if not self.effective_approx_ratio:
            raise ValueError("effective_approx_ratio must be populated to compute score.")
        return BenchmarkScore(value=float(np.mean(self.effective_approx_ratio)))


def prepare_qaoa_circuit(
    graph: nx.Graph, qaoa_layers: list, graph_type: GraphType, circuit_encoding: EncodingType
) -> list[QuantumCircuit]:
    """Prepare a list of QAOA circuits for the given graph and qaoa_layers.
    Args:
        graph: Networkx graph of the problem.
        qaoa_layers: List of p-layer values for the QAOA circuit.
        graph_type: Type of graph used in the experiment (1D, NL, FC).
        circuit_encoding: if connectivity of the device graph is not fully connected and the test is FC, it is used the SWAP encoding.
    Returns:
        List of QAOA circuits for each p_layer.
    """
    circuits = []
    for layer_p_i in qaoa_layers:
        circuit = qaoa_circuit(graph, layer_p_i, graph_type, circuit_encoding)
        circuits.append(circuit)
    return circuits


@dataclass
class TrialStats:
    """Data class to store statistics of a single trial.

    Attributes:
        qubits: Number of qubits used in the circuit.
        approx_ratio: Approximation ratio score.
        optimal_probability: probability of measuring optimal solution.
        p_value: p-value for the LR-QAOA 1D count.
        confidence_level: Confidence level for benchmarking.
        confidence_pass: Boolean indicating if the r-value is below the confidence level.
        num_random_trials: Number of random trials used to calculate the p-value.
    """

    qubits: int
    approx_ratio: float
    optimal_probability: float
    p_value: float
    confidence_pass: bool


@dataclass
class AggregateStats:
    """Data class to store aggregated statistics over multiple trials.

    Attributes:
        trial_stats: List of TrialStats objects for each trial.
        trials: Number of trials aggregated.
        optimal_probability: Average probability of measuring optimal solution.
        approx_ratio: Average approx_ratio-value for all trials.
        p_value: Combined p-value for all trials.
        approx_ratio_pass: Boolean indicating whether all trials exceeded the r random threshold.
        confidence_pass: Boolean indicating if all trials passed the confidence level.
    """

    trial_stats: list[list[TrialStats]]
    trials: int
    optimal_probability: list[float]
    approx_ratio: list[float]
    confidence_pass: list[bool]
    effective_approx_ratio: list[float]


def calc_trial_stats(
    num_qubits: int,
    graph_info: list[list],
    optimal_sol: str,
    samples: dict[str, int],
    confidence_level: float,
    num_random_trials: int,
    approx_ratio_random_mean: float,
    approx_ratio_random_std: float,
) -> TrialStats:
    """Calculate various statistics for linear ramp QAOA benchmarking.

    Args:
        graph_info: problem edges and weights [[u1,u2,w12], [u1,u3,w13],...] where u1, u2, and u3 are nodes and w13 is the weight between nodes u1 and u3.
        optimal_sol: Optimal solution bitstring for the problem.
        samples: A dictionary of bitstrings to counts measured from the backend.
        confidence_level: Specified confidence level for the benchmarking.
        approx_ratio_random_mean: mean value of the performance of the random sampler
        approx_ratio_random_std: standard deviation value of the performance of the random sampler
    Returns:
        A `TrialStats` object containing the calculated statistics.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    graph.add_weighted_edges_from(graph_info)
    results_qpu = objective_func(samples, graph, optimal_sol)
    approx_ratio = results_qpu["approx_ratio"]
    t_score = (approx_ratio - approx_ratio_random_mean) / approx_ratio_random_std
    p_val = float(1 - stats.t.cdf(t_score, df=num_random_trials - 1))

    return TrialStats(
        qubits=num_qubits,
        approx_ratio=approx_ratio,
        optimal_probability=results_qpu["optimal_probability"],
        p_value=p_val,
        confidence_pass=p_val < confidence_level,
    )


def calc_stats(data: LinearRampQAOAData, samples: list["MeasCount"]) -> AggregateStats:
    """Calculate aggregate statistics over multiple trials.

    Args:
        data: contains dispatch-time data (input data + random statistics).
        samples: contains results from the quantum device.
    Returns:
        An AggregateStats object containing aggregated statistics for the result.
    """
    trial_stats = []

    num_trials = data.trials
    # Process each trial, handling provider-specific logic.
    num_circ = 0
    for trial in range(num_trials):
        trial_p = []
        for num_qaoa_layers in data.qaoa_layers:
            trial_stat = calc_trial_stats(
                num_qubits=data.num_qubits,
                graph_info=data.graph_info,
                optimal_sol=data.optimal_sol,
                samples=samples[num_circ],
                confidence_level=data.confidence_level,
                num_random_trials=data.num_random_trials,
                approx_ratio_random_mean=data.approx_ratio_random_mean,
                approx_ratio_random_std=data.approx_ratio_random_std,
            )
            trial_p.append(trial_stat)
            num_circ += 1
        trial_stats.append(trial_p)

    # Aggregate the trial statistics.
    optimal_probability = [
        sum(stat[ith_layer].optimal_probability for stat in trial_stats) / num_trials
        for ith_layer in range(len(data.qaoa_layers))
    ]
    approx_ratio = [
        sum(stat[ith_layer].approx_ratio for stat in trial_stats) / num_trials
        for ith_layer in range(len(data.qaoa_layers))
    ]
    confidence_pass = [
        all(stat[ith_layer].confidence_pass for stat in trial_stats)
        for ith_layer in range(len(data.qaoa_layers))
    ]
    effective_approx_ratio = [
        (r - data.approx_ratio_random_mean) / (1 - data.approx_ratio_random_mean)
        for r in approx_ratio
    ]

    return AggregateStats(
        trial_stats=trial_stats,
        trials=num_trials,
        approx_ratio=approx_ratio,
        optimal_probability=optimal_probability,
        confidence_pass=confidence_pass,
        effective_approx_ratio=effective_approx_ratio,
    )


class LinearRampQAOA(Benchmark):
    def _build_circuits(
        self, device: "QuantumDevice"
    ) -> tuple[list[QuantumCircuit], list[tuple[int, int, float]], str, EncodingType]:
        """Shared circuit construction logic.

        Args:
            device: The quantum device to build circuits for.

        Returns:
            Tuple of (circuits_with_params, graph_info, optimal_sol, circuit_encoding).
        """
        num_qubits = self.params.num_qubits
        graph_type = self.params.graph_type
        qaoa_layers = self.params.qaoa_layers
        trials = self.params.trials
        delta_beta = self.params.delta_beta
        delta_gamma = self.params.delta_gamma
        seed = self.params.seed

        random.seed(seed)  # set seed for reproducibility
        if device.id == "aer_simulator" and graph_type == "NL":
            graph_device = rx.generators.star_graph(num_qubits)
        else:
            graph_device = connectivity_graph(device)
        edges_device = list(graph_device.edge_list())
        circuit_encoding: EncodingType = "Direct"

        if graph_type == "1D":
            edges = [(i, i + 1) for i in range(num_qubits - 1)]
        elif graph_type == "NL":
            num_qubits_device = graph_device.num_nodes()
            if num_qubits != num_qubits_device:
                raise ValueError(
                    f"Number of qubits ({num_qubits}) does not match the device's number of qubits ({num_qubits_device})."
                )
            edges = edges_device
            if len(edges) == num_qubits * (num_qubits - 1) / 2:
                raise TypeError(
                    "The device is a fully connected device. Implement the graph_type 'FC' test."
                )
        elif graph_type == "FC":
            edges = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
            if not all(edge in edges_device for edge in edges):
                # in case the quantum device is not fully connected the SWAP networks encoding is implemented.
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
        optimal_sol = weighted_maxcut_solver(graph)

        circuits = prepare_qaoa_circuit(
            graph=graph,
            qaoa_layers=qaoa_layers,
            graph_type=graph_type,
            circuit_encoding=circuit_encoding,
        )

        circuits_with_params = []
        for trial_i in range(trials):
            for p_layer_i, circuit in zip(qaoa_layers, circuits):
                linear_ramp = list(range(1, p_layer_i + 1))
                betas = [i * delta_beta / p_layer_i for i in reversed(linear_ramp)]
                gammas = [i * delta_gamma / p_layer_i for i in linear_ramp]
                circuits_with_params.append(circuit.assign_parameters(betas + gammas))

        return circuits_with_params, graph_info, optimal_sol, circuit_encoding

    def dispatch_handler(self, device: "QuantumDevice") -> LinearRampQAOAData:
        circuits_with_params, graph_info, optimal_sol, circuit_encoding = self._build_circuits(
            device
        )

        approx_ratio_random_mean, approx_ratio_random_std = calc_random_stats(
            self.params.num_qubits,
            graph_info,
            self.params.shots,
            self.params.num_random_trials,
            optimal_sol,
        )

        return LinearRampQAOAData.from_quantum_job(
            quantum_job=device.run(circuits_with_params, shots=self.params.shots),
            num_qubits=self.params.num_qubits,
            graph_info=graph_info,
            optimal_sol=optimal_sol,
            shots=self.params.shots,
            confidence_level=self.params.confidence_level,
            trials=self.params.trials,
            num_random_trials=self.params.num_random_trials,
            seed=self.params.seed,
            qaoa_layers=self.params.qaoa_layers,
            delta_beta=self.params.delta_beta,
            delta_gamma=self.params.delta_gamma,
            graph_type=self.params.graph_type,
            circuit_encoding=circuit_encoding,
            approx_ratio_random_mean=approx_ratio_random_mean,
            approx_ratio_random_std=approx_ratio_random_std,
        )

    def poll_handler(
        self,
        job_data: LinearRampQAOAData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> LinearRampQAOAResult:
        stats: AggregateStats = calc_stats(job_data, flatten_counts(result_data))
        return LinearRampQAOAResult(
            approx_ratio=stats.approx_ratio,
            random_approx_ratio=job_data.approx_ratio_random_mean,
            confidence_pass=stats.confidence_pass,
            effective_approx_ratio=stats.effective_approx_ratio,
        )

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list["CircuitBatch"]:
        circuits_with_params, _, _, _ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits_with_params, shots=self.params.shots)]
