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

from qc_app_oriented_benchmarks.bernstein_vazirani.bv_benchmark import run

import types


@dataclass
class BernsteinVaziraniResult(BenchmarkResult):
    """Stores the results from running Bernstein-Vazirani Benchmark.
    Results:
        final_metrics: A QED-C metrics object containing final results.
    """

    final_metrics: types.ModuleType


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
    metrics: types.ModuleType


def analyze_results(metrics: types.ModuleType, counts_list: list[MeasCount]) -> None:
    """
    Iterates over each circuit group and secret int to compute the fidelities.

    Args:
        metrics: A QED-C metrics module object.
        counts_list: A list of all counts objects, each index corresponds to a circuit.

    Returns:
        None: the modification is done in-place and stored in the metrics object.
    """

    info: dict[str, dict[str, dict[str, float]]] = metrics.circuit_metrics

    num_qubits_list: list[str] = list(info.keys())

    counts_idx: int = 0

    for num_qubits in num_qubits_list:
        s_str_list = list(info[num_qubits].keys())

        for s_str in s_str_list:
            counts: dict[str, int] = counts_list[counts_idx]

            counts_idx += 1

            correct_dist = {format(int(s_str), f"0{int(num_qubits) - 1}b"): 1.0}

            fidelity = metrics.polarization_fidelity(counts, correct_dist)

            metrics.store_metric(num_qubits, s_str, "fidelity", fidelity)

    # The metrics object now has all the fidelities stored.


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
            input_value=None,
            backend_id=None,
            provider_backend=None,
            hub="ibm-q",
            group="open",
            project="main",
            exec_options=None,
            context=None,
            api=None,
            get_circuits=True,
        )

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
        metrics = job_data.metrics

        counts_list = flatten_counts(result_data)

        analyze_results(metrics, counts_list)

        metrics.aggregate_metrics()

        # For now, try and plot with just method 1, worry about method 2 later.
        metrics.plot_metrics(
            "Benchmark Results - Bernstein-Vazirani (1) - Qiskit", filters=["fidelity"]
        )

        return BernsteinVaziraniResult(final_metrics=metrics)
