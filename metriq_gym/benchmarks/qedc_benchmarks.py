"""
A general structure for dispatching and polling QED-C benchmarks.
Credit to QED-C for implementing the benchmarks.

The benchmarks generate N circuits for M qubits ranging from min_qubits to max_qubits.
Each circuit is then run, and the metrics are computed.
"""

from dataclasses import dataclass

from qbraid import GateModelResultData, QuantumDevice, QuantumJob

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts

from metriq_gym.helpers.qedc_helpers import QEDC_Metrics, analyze_results, get_circuits_and_metrics

from qiskit import QuantumCircuit


@dataclass
class QEDCData(BenchmarkData):
    """
    Stores the input parameters or metadata for a QED-C benchmark.
    Note that any benchmark specific parameters will be added on to these.

    Parameters/Metadata:
        num_shots: number of shots for each circuit to be ran with.
        min_qubits: minimum number of qubits to start generating circuits for the benchmark.
        max_qubits: maximum number of qubits to stop generating circuits for the benchmark.
        skip_qubits: the step size for generating circuits from the min to max qubit sizes.
        max_circuits: maximum number of circuits generated for each qubit size in the benchmark.
        circuit_metrics: stores QED-C circuit creation metrics data.
        circuits: the list of quantum circuits ran, it's needed to poll the results with QED-C.
        circuit_identifiers: the unique identifiers for circuits (num qubits, secret str),
                             used to preserve order when polling.
        benchmark_name: the name of the benchmark being ran.
        method: which QED-C method to run the benchmark with.
    """

    circuit_metrics: QEDC_Metrics
    circuits: list[QuantumCircuit]
    circuit_identifiers: list[tuple[str, str]]


class QEDCResult(BenchmarkResult):
    """
    Stores the results from running a QED-C benchmark.

    Results:
        circuit_metrics: Stores all QED-C metrics to output.
    """

    circuit_metrics: QEDC_Metrics


class QEDCBenchmarks(Benchmark):
    """Benchmark class for QED-C experiments."""

    def dispatch_handler(self, device: QuantumDevice) -> QEDCData:
        # For more information on the parameters, view the schema for this benchmark.
        num_shots = self.params.num_shots

        circuits, circuit_metrics, circuit_identifiers = get_circuits_and_metrics(
            params=self.params.model_dump(),
        )

        return QEDCData.from_quantum_job(
            quantum_job=device.run(circuits, shots=num_shots),
            circuit_metrics=circuit_metrics,
            circuits=circuits,
            circuit_identifiers=circuit_identifiers,
        )

    def poll_handler(
        self,
        job_data: QEDCData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> QEDCResult:
        counts_list = flatten_counts(result_data)

        # Call the QED-C method after some pre-processing to obtain metrics.
        circuit_metrics = analyze_results(self.params.model_dump(), job_data, counts_list)

        return QEDCResult(circuit_metrics=circuit_metrics)
