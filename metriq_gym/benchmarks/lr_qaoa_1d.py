import math
import statistics
import networkx as nx
from scipy import stats
from dataclasses import dataclass

from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount
from pyqrack import QrackSimulator
from qiskit import QuantumCircuit

from metriq_gym.circuits import qaoa_1d_circuit

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.helpers.lrqaoa_helpers import objective_func, random_samples

@dataclass
class LRQAOA1dData(BenchmarkData):
    num_qubits: int
    shots: int
    p: int
    r: float

@dataclass
class LRQAOA1dResult(BenchmarkResult):
    num_qubits: int
    r: float

def prepare_lrqaoa_1d_circuit(nq: int, p_layers: list) -> list[QuantumCircuit]:
    G = nx.Graph()
    with open("../data/lr_qaoa_1d.json", "r") as data:
        if nq in data:
            G.add(data[nq]["problem"])
        else:
            raise ValueError(f"Graph for nq={nq} not found in the data file.")
    circuits = []
    for p in p_layers:
        circuit = qaoa_1d_circuit(G, p)
        circuit.measure_all()
        circuits.append(circuit)
    return circuits


@dataclass
class TrialStats:
    """Data class to store statistics of a single trial.

    Attributes:
        qubits: Number of qubits used in the circuit.
        shots: Number of measurement shots performed on the quantum circuit.
        r: Approximation ratio score.
        prob: Probability of measuring optimal solution.
        p_value: p-value for the LR-QAOA 1D count.
        confidence_level: Confidence level for benchmarking.
        confidence_pass: Boolean indicating if the r-value is below the confidence level.
    """

    qubits: int
    shots: int
    r: float
    prob: float
    p_value: float
    confidence_pass: bool


@dataclass
class AggregateStats:
    """Data class to store aggregated statistics over multiple trials.

    Attributes:
        trial_stats: List of TrialStats objects for each trial.
        trials: Number of trials aggregated.
        prob: Average probability of measuring optimal solution.
        r: Combined r-value for all trials.
        p_value: Combined p-value for all trials.
        r_pass: Boolean indicating whether all trials exceeded the r random threshold.
        confidence_pass: Boolean indicating if all trials passed the confidence level.
    """

    trial_stats: list[TrialStats]
    trials: int
    prob: float
    r: float
    r_pass: bool
    p_value: float
    confidence_pass: bool

def calc_trial_stats(
    nq: int,
    counts: dict[str, int],
    confidence_level: float,
    num_random_trials: int,
) -> TrialStats:
    """Calculate various statistics for quantum volume benchmarking.

    Args:
        ideal_probs: A dictionary of bitstrings to ideal probabilities.
        counts: A dictionary of bitstrings to counts measured from the backend.
        shots: Number of measurement shots performed on the quantum circuit.
        confidence_level: Specified confidence level for the benchmarking.

    Returns:
        A `TrialStats` object containing the calculated statistics.
    """
    G = nx.Graph()
    with open("../data/lr_qaoa_1d.json", "r") as data:
        if nq in data:
            G.add_weighted_edges_from(data[nq]["problem"])
            optimal_sol = data[nq]["optimal"]
        else:
            raise ValueError(f"Graph for nq={nq} not found in the data file.")
    shots = sum(counts.values()) 
    random_samples_dict = random_samples(shots, len(G.nodes()))
    r_random_list = []
    for i in range(num_random_trials):
        r_random_list.append(objective_func(random_samples_dict, G, optimal_sol)["r"])
    r_random_mean = statistics.mean(r_random_list)
    r_random_std = statistics.std(r_random_list)
    results_qpu = objective_func(counts, G, optimal_sol)
    r = results_qpu["r"]
    t_score = (r - r_random_mean) / r_random_std
    p_val = (1 - stats.t.cdf(t_score, df=num_random_trials-1))

    return TrialStats(
        qubits=nq,
        shots=shots,
        r=r,
        probability=results_qpu["probability"],
        r_random=r_random_mean,
        p_value=p_val,
        confidence_pass=p_val < confidence_level,
    )


def calc_stats(data: LRQAOA1dData, counts: list[MeasCount]) -> AggregateStats:
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
            nq=data.num_qubits,
            counts=counts[trial],
            confidence_level=data.confidence_level,
            num_random_trials=data.num_random_trials
        )
        trial_stats.append(trial_stat)

    # Aggregate the trial statistics.
    probability = sum(stat.probability for stat in trial_stats) / num_trials
    r  = sum(stat.r for stat in trial_stats) / num_trials
    p_value = math.prod(stat.p_value for stat in trial_stats) ** (1 / num_trials)

    return AggregateStats(
        trial_stats=trial_stats,
        trials=num_trials,  # Set the trials field to fix the type error
        r=r,
        probability=probability,
        p_value=p_value,
        confidence_pass=all(stat.confidence_pass for stat in trial_stats),
    )

class LRQAOA1D(Benchmark):
    def dispatch_handler(self, device: QuantumDevice) -> LRQAOA1dData:
        num_qubits = self.params.num_qubits
        p_layers = self.params.p # Number of QAOA layers
        shots = self.params.shots
        trials = self.params.trials
        betas = self.params.betas
        gammas = self.params.gammas
        if len(betas) != len(gammas):
            raise ValueError("Length of betas and gammas must be equal.")
        circuits = prepare_lrqaoa_1d_circuit(n=num_qubits, p=p_layers)
        circuits = [circuit.assign_parameters((betas, gammas)) for circuit in circuits]
        quantum_job: QuantumJob | list[QuantumJob] = device.run(circuits, shots=shots)
        provider_job_ids = (
            [quantum_job.id]
            if isinstance(quantum_job, QuantumJob)
            else [job.id for job in quantum_job]
        )
        return LRQAOA1dData(
            provider_job_ids=provider_job_ids,
            num_qubits=num_qubits,
            shots=shots,
            depth=num_qubits,
            confidence_level=self.params.confidence_level,
            trials=trials,
        )

    def poll_handler(
        self,
        job_data: LRQAOA1dData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> LRQAOA1dResult:
        stats: AggregateStats = calc_stats(job_data, flatten_counts(result_data))

        return LRQAOA1dResult(
            num_qubits=job_data.num_qubits,
            p_layers=job_data.p_layers,
            r=stats.r,
            probability=stats.probability,
            confidence_pass=stats.confidence_pass,
            p_value=stats.p_value,
        )
