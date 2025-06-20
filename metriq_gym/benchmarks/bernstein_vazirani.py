"""
Bernstein-Vazirani Benchmark for metriq-gym
Credit to QED-C for implementing the benchmark.

The benchmark generates N circuits for X qubits ranging from min_qubits to max_qubits.
Each circuit is then ran, and the metrics are computed.
"""

from dataclasses import dataclass

from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts

from qedc.bernstein_vazirani.bv_benchmark import run
from qedc._common import metrics as qedc_metrics


@dataclass
class BernsteinVaziraniResult(BenchmarkResult):
    """Stores the results from running Bernstein-Vazirani Benchmark.
    Results:
        fidelity: Stores QED-C fidelity calculations
        fidelity_plot: To be added
    """

    fidelity: list[dict[str, float]]


@dataclass
class BernsteinVaziraniData(BenchmarkData):
    """Stores the input parameters or metadata for Bernstein-Vazirani Benchmark.
    Paramters/Metadata:
        shots: number of shots for each circuit to be ran with.
        min_qubits: minimum number of qubits to start generating circuits for the benchmark.
        max_qubits: maximum number of qubits to stop generating circuits for the benchmark.
        skip_qubits: the step size for generating circuits from the min to max qubit sizes.
        max_circuits: maximum number of circuits generated for each qubit size in the benchmark.
        metrics: Stores QED-C circuit metrics data.
    """

    shots: int
    min_qubits: int
    max_qubits: int
    skip_qubits: int
    max_circuits: int
    metrics: dict[str, dict[str, dict[str, float]]]


def analyze_results(
    metrics: dict[str, dict[str, dict[str, float]]], counts_list: list[MeasCount]
) -> list[dict[str, float]]:
    """
    Iterates over each circuit group and secret int to compute the fidelities.

    Args:
        metrics: Stored QED-C circuit metrics
        counts_list: A list of all counts objects, each index corresponds to a circuit.

    Returns:
        fidelity: Stores QED-C fidelity calculations with respect to circuits in the counts_list.
    """

    info: dict[str, dict[str, dict[str, float]]] = metrics

    num_qubits_list: list[str] = list(info.keys())

    counts_idx: int = 0

    all_fidelity = []

    for num_qubits in num_qubits_list:
        s_str_list = list(info[num_qubits].keys())

        for s_str in s_str_list:
            counts: dict[str, int] = counts_list[counts_idx]

            counts_idx += 1

            correct_dist = {format(int(s_str), f"0{int(num_qubits) - 1}b"): 1.0}

            fidelity = qedc_metrics.polarization_fidelity(counts, correct_dist)

            all_fidelity.append(fidelity)

    return all_fidelity


class BernsteinVazirani(Benchmark):
    """Benchmark class for Bernstein-Vazirani experiments."""

    def dispatch_handler(self, device: QuantumDevice) -> BernsteinVaziraniData:
        # For more information on the parameters, view the schema for this benchmark.
        shots = self.params.shots
        min_qubits = self.params.min_qubits
        max_qubits = self.params.max_qubits
        skip_qubits = self.params.skip_qubits
        max_circuits = self.params.max_circuits

        # Call the QED-C submodule to get the circuits and creation information.
        circuits, metrics = run(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            skip_qubits=skip_qubits,
            max_circuits=max_circuits,
            num_shots=shots,
            method=1,
            get_circuits=True,
        )

        # Store job IDs
        quantum_job: QuantumJob | list[QuantumJob] = device.run(circuits, shots=shots)
        provider_job_ids = (
            [quantum_job.id]
            if isinstance(quantum_job, QuantumJob)
            else [job.id for job in quantum_job]
        )

        return BernsteinVaziraniData(
            provider_job_ids=provider_job_ids,
            shots=shots,
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            skip_qubits=skip_qubits,
            max_circuits=max_circuits,
            metrics=metrics.circuit_metrics,
        )

    def poll_handler(
        self,
        job_data: BernsteinVaziraniData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> BernsteinVaziraniResult:
        metrics = job_data.metrics

        counts_list = flatten_counts(result_data)

        fidelity = analyze_results(metrics, counts_list)

        return BernsteinVaziraniResult(fidelity=fidelity)
