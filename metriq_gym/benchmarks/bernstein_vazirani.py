"""
Bernstein-Vazirani Benchmark for metriq-gym
Credit to QED-C for implementing the benchmark.

The benchmark generates N circuits for X qubits ranging from min_qubits to max_qubits.
Each circuit is then ran, and the metrics are computed.
"""

from dataclasses import dataclass

from qbraid import GateModelResultData, QuantumDevice, QuantumJob

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult


@dataclass
class BernsteinVaziraniResult(BenchmarkResult):
    """Stores the results from running Bernstein-Vazirani Benchmark.
    Results:
        final_metrics: A QED-C metrics object containing final results.
    """

    # Note: format of this might change after using the QED-C module and computing
    #       final metrics; this is a place holder for now.
    # final_metrics: dict[str, dict[str, dict[str, str]]]
    final_metrics: None


@dataclass
class BernsteinVaziraniData(BenchmarkData):
    """Stores the input parameters or metadata for Bernstein-Vazirani Benchmark.
    Paramters/Metadata:
        shots: number of shots for each circuit to be ran with.
        min_qubits: minimum number of qubits to start generating circuits for the benchmark.
        max_qubits: maximum number of qubits to stop generating circuits for the benchmark.
        skip_qubits: the step size for generating circuits from the min to max qubit sizes.
        max_circuits: maximum number of circuits generated for each qubit size in the benchmark.
        metrics: QED-C returned object storing creation information, it will be used to process results.
    """

    shots: int
    min_qubits: int
    max_qubits: int
    skip_qubits: int
    max_circuits: int
    # metrics: dict[str, dict[str, dict[str, str]]]
    metrics: None


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
        circuits, metrics = None, None  # Temprorary assignment, work in progress.

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
            metrics=metrics,
        )

    def poll_handler(
        self,
        job_data: BernsteinVaziraniData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> BernsteinVaziraniResult:
        # To implement: calling QED-C metrics module and using the info to create final metrics.
        metrics = job_data.metrics
        final_metrics = metrics  # Temprorary assignment, work in progress.

        return BernsteinVaziraniResult(final_metrics=final_metrics)
