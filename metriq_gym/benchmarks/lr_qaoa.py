import math
import statistics
import networkx as nx
from scipy import stats
from dataclasses import dataclass
import numpy as np

from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount
from qiskit import QuantumCircuit

from metriq_gym.circuits import qaoa_circuit

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.helpers.lrqaoa_helpers import weighted_maxcut_solver, objective_func, random_samples

from typing import Literal

GraphType = Literal["1D", "NL", "FC"]


@dataclass
class LinearRampQAOAData(BenchmarkData):
    provider_job_ids: list[str]
    num_qubits: int
    graph: nx.Graph
    graph_type: GraphType
    optimal_sol: str
    trials: int
    num_random_trials: int
    confidence_level: float
    shots: int
    p_layers: int
    seed: int
    delta_beta: float
    delta_gamma: float


@dataclass
class LinearRampQAOAResult(BenchmarkResult):
    num_qubits: int
    approx_ratio: float
    probability: float
    confidence_pass: bool
    p_value: float
    trails: int


def prepare_qaoa_circuit(
    graph: nx.Graph, p_layers: list, graph_type: GraphType
) -> list[QuantumCircuit]:
    """Prepare a list of QAOA circuits for the given graph and p_layers.
    Args:
        graph: Networkx graph of the problem.
        p_layers: List of p-layer values for the QAOA circuit.
        graph_type: Type of graph used in the experiment (1D, NL, FC).
    Returns:
        List of QAOA circuits for each p_layer.
    """
    circuits = []
    for layer_p_i in p_layers:
        circuit = qaoa_circuit(graph, layer_p_i, graph_type)
        circuit.measure_all()
        circuits.append(circuit)
    return circuits


@dataclass
class TrialStats:
    """Data class to store statistics of a single trial.

    Attributes:
        qubits: Number of qubits used in the circuit.
        shots: Number of measurement shots performed on the quantum circuit.
        approx_ratio: Approximation ratio score.
        approx_ratio_random: Approximation ratio score of a random sampler.
        prob: Probability of measuring optimal solution.
        p_value: p-value for the LR-QAOA 1D count.
        confidence_level: Confidence level for benchmarking.
        confidence_pass: Boolean indicating if the r-value is below the confidence level.
        num_random_trials: Number of random trials used to calculate the p-value.
    """

    qubits: int
    shots: int
    approx_ratio: float
    approx_ratio_random: float
    probability: float
    p_value: float
    confidence_pass: bool
    num_random_trials: int


@dataclass
class AggregateStats:
    """Data class to store aggregated statistics over multiple trials.

    Attributes:
        trial_stats: List of TrialStats objects for each trial.
        trials: Number of trials aggregated.
        probability: Average probability of measuring optimal solution.
        approx_ratio: Average approx_ratio-value for all trials.
        approx_ratio_random: Average approx_ratio-value of the random sampler for all trials.
        p_value: Combined p-value for all trials.
        approx_ratio_pass: Boolean indicating whether all trials exceeded the r random threshold.
        confidence_pass: Boolean indicating if all trials passed the confidence level.
        num_random_trials: Number of random trials used to calculate the p-value.
    """

    trial_stats: list[TrialStats]
    trials: int
    probability: float
    approx_ratio: float
    approx_ratio_random: float
    p_value: float
    confidence_pass: bool
    num_random_trials: int


def calc_trial_stats(
    graph: nx.graph,
    optimal_sol: str,
    counts: dict[str, int],
    shots: int,
    confidence_level: float,
    num_random_trials: int,
) -> TrialStats:
    """Calculate various statistics for linear ramp QAOA benchmarking.

    Args:
        graph: Networkx graph of the problem.
        optimal_sol: Optimal solution bitstring for the problem.
        graph_type: Type of graph used in the experiment.
        seed": Seed for graph and the random number generator.
        counts: A dictionary of bitstrings to counts measured from the backend.
        shots: Number of measurement shots performed on the quantum circuit.
        confidence_level: Specified confidence level for the benchmarking.
        num_random_trials: random sampler number of trails

    Returns:
        A `TrialStats` object containing the calculated statistics.
    """
    nq = graph.number_of_nodes()  # number of qubits
    shots = sum(counts.values())
    random_samples_dict = random_samples(shots, len(graph.nodes()))
    approx_ratio_random_list = []
    for _ in range(num_random_trials):
        approx_ratio_random_list.append(
            objective_func(random_samples_dict, graph, optimal_sol)["r"]
        )
    approx_ratio_random_mean = statistics.mean(approx_ratio_random_list)
    approx_ratio_random_std = statistics.pstdev(approx_ratio_random_list)
    results_qpu = objective_func(counts, graph, optimal_sol)
    approx_ratio = results_qpu["approx_ratio"]
    t_score = (approx_ratio - approx_ratio_random_mean) / approx_ratio_random_std
    p_val = 1 - stats.t.cdf(t_score, df=num_random_trials - 1)

    return TrialStats(
        qubits=nq,
        shots=shots,
        num_random_trials=num_random_trials,
        approx_ratio=approx_ratio,
        approx_ratio_random=approx_ratio_random_mean,
        probability=results_qpu["probability"],
        p_value=p_val,
        confidence_pass=p_val < confidence_level,
    )


def calc_stats(data: LinearRampQAOAData, counts: list[MeasCount]) -> AggregateStats:
    """Calculate aggregate statistics over multiple trials.

    Args:
        data: contains dispatch-time data (input data + ideal probability).
        counts: contains results from the quantum device (one MeasCount per trial).
    Returns:
        An AggregateStats object containing aggregated statistics for the result.
    """
    trial_stats = []

    num_trials = len(counts)
    # Process each trial, handling provider-specific logic.
    for trial in range(num_trials):
        trial_stat = calc_trial_stats(
            graph=data.graph,
            optimal_sol=data.optimal_sol,
            counts=counts[trial],
            shots=data.shots,
            confidence_level=data.confidence_level,
            num_random_trials=data.num_random_trials,
        )
        trial_stats.append(trial_stat)

    # Aggregate the trial statistics.
    probability = sum(stat.probability for stat in trial_stats) / num_trials
    approx_ratio = sum(stat.approx_ratio for stat in trial_stats) / num_trials
    approx_ratio_random = sum(stat.approx_ratio_random for stat in trial_stats) / num_trials
    p_value = math.prod(stat.p_value for stat in trial_stats) ** (1 / num_trials)

    return AggregateStats(
        trial_stats=trial_stats,
        trials=num_trials,
        num_random_trials=data.num_random_trials,
        approx_ratio=approx_ratio,
        approx_ratio_random=approx_ratio_random,
        probability=probability,
        p_value=p_value,
        confidence_pass=all(stat.confidence_pass for stat in trial_stats),
    )


class LinearRampQAOA(Benchmark):
    def dispatch_handler(self, device: QuantumDevice) -> LinearRampQAOAData:
        num_qubits = self.params.num_qubits
        graph_type = self.params.graph_type
        p_layers = self.params.p
        shots = self.params.shots
        trials = self.params.trials
        num_random_trials = self.params.num_random_trials
        delta_beta = self.params.delta_beta
        delta_gamma = self.params.delta_gamma
        seed = self.params.seed
        confidence_level = self.params.confidence_level

        np.random.seed(seed)  # set seed for reproducibility

        if graph_type == "1D":
            edges = [(i, i + 1) for i in range(num_qubits - 1)]
        elif graph_type == "NL":
            graph_device = device.coupling_map.graph.to_undirected()
            num_qubits_device = graph_device.num_nodes()
            if num_qubits != num_qubits_device:
                raise ValueError(
                    f"Number of qubits ({num_qubits}) does not match the device's number of qubits ({num_qubits_device})."
                )
            edges = list(graph_device.edge_list())
        elif graph_type == "FC":
            edges = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        else:
            raise ValueError(
                f"Unsupported graph type: {graph_type}. Supported types are '1D', 'NL', and 'FC'."
            )

        graph = nx.Graph()
        graph.add_nodes_from(range(num_qubits))
        graph.add_weighted_edges_from(
            [(i, j, np.random.choice([0.1, 0.2, 0.3, 0.5, 1.0])) for i, j in edges]
        )
        optimal_sol = weighted_maxcut_solver(graph)
        circuits = prepare_qaoa_circuit(graph=graph, p_layers=p_layers, graph_type=graph_type)
        circuits_with_params = []
        for pi, circuit in zip(p_layers, circuits):
            betas = np.arange(pi, 0, -1) * delta_beta / pi
            gammas = np.arange(1, pi + 1) * delta_gamma / pi
            circuits_with_params.append(
                circuit.assign_parameters((betas, gammas))
            )  # assing linear ramp parameters
        quantum_job: QuantumJob | list[QuantumJob] = device.run(circuits_with_params, shots=shots)
        provider_job_ids = (
            [quantum_job.id]
            if isinstance(quantum_job, QuantumJob)
            else [job.id for job in quantum_job]
        )
        return LinearRampQAOAData(
            provider_job_ids=provider_job_ids,
            num_qubits=num_qubits,
            graph=graph,
            optimal_sol=optimal_sol,
            shots=shots,
            confidence_level=confidence_level,
            trials=trials,
            num_random_trials=num_random_trials,
            seed=seed,
            p_layers=p_layers,
            delta_beta=delta_beta,
            delta_gamma=delta_gamma,
            graph_type=graph_type,
        )

    def poll_handler(
        self,
        job_data: LinearRampQAOAData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> LinearRampQAOAResult:
        stats: AggregateStats = calc_stats(job_data, flatten_counts(result_data))

        return LinearRampQAOAResult(
            num_qubits=job_data.num_qubits,
            p_layers=job_data.p_layers,
            approx_ratio=stats.approx_ratio,
            probability=stats.probability,
            confidence_pass=stats.confidence_pass,
            p_value=stats.p_value,
        )
