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

from qedc.bernstein_vazirani.bv_benchmark import run, analyze_and_print_result, qedc_benchmarks_init
from qedc._common import metrics

from qiskit import QuantumCircuit


class BernsteinVaziraniResult(BenchmarkResult):
    """Stores the results from running Bernstein-Vazirani Benchmark.
    Results:
        circuit_metrics: Stores all QED-C metrics to output.
    """

    circuit_metrics: dict[str, dict[str, dict[str, float]]]


@dataclass
class BernsteinVaziraniData(BenchmarkData):
    """Stores the input parameters or metadata for Bernstein-Vazirani Benchmark.
    Paramters/Metadata:
        shots: number of shots for each circuit to be ran with.
        min_qubits: minimum number of qubits to start generating circuits for the benchmark.
        max_qubits: maximum number of qubits to stop generating circuits for the benchmark.
        skip_qubits: the step size for generating circuits from the min to max qubit sizes.
        max_circuits: maximum number of circuits generated for each qubit size in the benchmark.
        circuit_metrics: stores QED-C circuit creation metrics data.
        circuits: the list of quantum circuits ran, it's needed to poll the results with QED-C.
        circuit_identifiers: the unique identifiers for circuits, used to preserve order when polling.
    """

    shots: int
    min_qubits: int
    max_qubits: int
    skip_qubits: int
    max_circuits: int
    circuit_metrics: dict[str, dict[str, dict[str, float]]]
    circuits: list[QuantumCircuit]
    circuit_identifiers: list[tuple[str, str]]


def analyze_results(
    job_data: BernsteinVaziraniData, counts_list: list[MeasCount]
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Iterates over each circuit group and secret int to process results.
    Uses QED-C submodule to obtain calculations.

    Args:
        job_data: The BernsteinVaziraniData object for the job.
        counts_list: A list of all counts objects, each index corresponds to a circuit.

    Returns:
        None
    """

    """
    A wrapper class to enable support with QED-C's method to analyze results. 
    """

    class CountsWrapper:
        def __init__(self, qc: QuantumCircuit, counts: dict[str, int]):
            self.qc = qc
            self.counts = counts

        def get_counts(self, qc):
            if qc == self.qc:
                return self.counts

    # Initialize metrics module in QED-C submodule.
    qedc_benchmarks_init()

    # Restore circuit metrics dictionary from the dispatch data
    metrics.circuit_metrics = job_data.circuit_metrics

    curr_idx: int = 0

    for num_qubits, s_str in job_data.circuit_identifiers:
        counts: dict[str, int] = counts_list[curr_idx]

        qc = job_data.circuits[curr_idx]

        resultObj = CountsWrapper(qc, counts)

        _, fidelity = analyze_and_print_result(
            qc, resultObj, int(num_qubits), int(s_str), job_data.shots
        )

        metrics.store_metric(int(num_qubits), int(s_str), "fidelity", fidelity)

        curr_idx += 1

    return metrics.circuit_metrics


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
        circuits, circuit_metrics = run(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            skip_qubits=skip_qubits,
            max_circuits=max_circuits,
            num_shots=shots,
            method=1,
            get_circuits=True,
        )

        del circuit_metrics["subtitle"]

        circuit_identifiers = []
        for num_qubits in circuit_metrics.keys():
            for s_str in circuit_metrics[num_qubits].keys():
                circuit_identifiers.append((num_qubits, s_str))

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
            circuit_metrics=circuit_metrics,
            circuits=circuits,
            circuit_identifiers=circuit_identifiers,
        )

    def poll_handler(
        self,
        job_data: BernsteinVaziraniData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> BernsteinVaziraniResult:
        counts_list = flatten_counts(result_data)

        circuit_metrics = analyze_results(job_data, counts_list)

        return BernsteinVaziraniResult(circuit_metrics=circuit_metrics)
