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


@dataclass
class QEDCData(BenchmarkData):
    """
    Stores metadata for a QED-C benchmark.

    Metadata:
        circuit_metrics: stores QED-C circuit creation metrics data.
        circuit_identifiers: the unique identifiers for circuits (num qubits, secret str),
                             used to preserve order when polling.
    """

    circuit_metrics: QEDC_Metrics
    circuit_identifiers: list[tuple[str, str]]


class QEDCResult(BenchmarkResult):
    """
    Stores the results from running a QED-C benchmark.

    Results:
        circuit_metrics: Stores all QED-C metrics to output.
    """

    circuit_metrics: QEDC_Metrics


class QEDCBenchmark(Benchmark):
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
