"""QED-C application-oriented benchmark wrapper.

Summary:
    Provides a generic dispatch/poll pipeline around the QED-C benchmark suite (Bernstein-
    Vazirani, Phase Estimation, Hidden Shift, Quantum Fourier Transform) via the QC-App-
    Oriented-Benchmarks submodule.

Result interpretation:
    Polling returns QEDCResult.circuit_metrics, a nested dictionary keyed by qubit count and
    circuit identifier, populated with the fidelity or related metrics computed by the QED-C
    analyser. Inspect the per-circuit entries to understand performance trends.

Reference:
    - QED-C QC-App-Oriented-Benchmarks repository for algorithm-specific methodology.
"""

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.constants import JobType
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.resource_estimation import CircuitBatch

from _common import metrics

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob
    from qbraid.runtime.result_data import MeasCount


QEDC_BENCHMARK_IMPORTS: dict[JobType, str] = {
    JobType.BERNSTEIN_VAZIRANI: "bernstein_vazirani.bv_benchmark",
    JobType.PHASE_ESTIMATION: "phase_estimation.pe_benchmark",
    JobType.HIDDEN_SHIFT: "hidden_shift.hs_benchmark",
    JobType.QUANTUM_FOURIER_TRANSFORM: "quantum_fourier_transform.qft_benchmark",
}

"""
Type: QEDC_Metrics
Description:
    The structure for all returned QED-C circuit metrics.
    The first key represents the number of qubits for the group of circuits.
    The second key represents the unique identifier for a circuit in the group.
        - This may be a secret string for Bernstein-Vazirani, theta value for Phase-Estimation,
          and so on. Benchmark specific documentation can be found in QED-C's
          QC-App-Oriented-Benchmarks repository.
    The third key represents the metric being stored.
Example for Bernstein-Vazirani:
{
'3':    {
        '1': {'create_time': 0.16371703147888184,
              'fidelity': 1.0,
              'hf_fidelity': 1.0},
        '2': {'create_time': 0.0005087852478027344,
              'fidelity': 1.0,
              'hf_fidelity': 1.0}
        },
'4':    {
        '1': {'create_time': 0.0005209445953369141,
              'fidelity': 1.0,
              'hf_fidelity': 1.0},
        '3': {'create_time': 0.00047206878662109375,
              'fidelity': 1.0,
              'hf_fidelity': 1.0},
        '5': {'create_time': 0.0005078315734863281,
              'fidelity': 1.0,
              'hf_fidelity': 1.0}
        }
}
"""
QEDC_Metrics = dict[str, dict[str, dict[str, float]]]


@dataclass
class QEDCData(BenchmarkData):
    """
    Stores metadata for a QED-C benchmark.

    Metadata:
        circuit_metrics: stores QED-C circuit creation metrics data.
        circuit_identifiers: the unique identifiers for circuits (num qubits, circuit id),
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

    def compute_score(self) -> float | None:
        return None


def import_benchmark_module(benchmark_name: str) -> ModuleType:
    """
    Import the correct module.

    Args:
        benchmark_name: the name of the benchmark being ran.
    """
    try:
        benchmark_enum = JobType(benchmark_name)
    except ValueError:
        raise ValueError(f"Invalid QED-C benchmark name: '{benchmark_name}'")

    if benchmark_enum not in QEDC_BENCHMARK_IMPORTS:
        raise ValueError(f"'{benchmark_name}' is not a valid QED-C benchmark.")

    module_name = QEDC_BENCHMARK_IMPORTS[benchmark_enum]

    return import_module(module_name)


def analyze_results(
    params: dict[str, float | str], job_data: QEDCData, counts_list: list["MeasCount"]
) -> QEDC_Metrics:
    """
    Iterates over each circuit group and circuit id to process results.
    Uses QED-C submodule to obtain calculations.

    Args:
        params: the parameters the benchmark ran with and benchmark_name.
        job_data: the QEDCData object for the job.
        counts_list: a list of all counts objects, each index corresponds to a circuit.

    Returns:
        circuit_metrics: the updated circuit metrics in QED-C's format.
    """

    class CountsWrapper:
        """
        A wrapper class to enable support with QED-C's method to analyze results.
        """

        def __init__(self, counts: dict[str, int]):
            self.counts = counts

        def get_counts(self, _):
            return self.counts

    # Import the correct module
    benchmark_name = str(params["benchmark_name"])
    benchmark = import_benchmark_module(benchmark_name)

    # Restore circuit metrics dictionary from the dispatch data
    metrics.circuit_metrics = job_data.circuit_metrics

    # Iterate and get the metrics for each circuit in the list.
    for curr_idx, counts in enumerate(counts_list):
        num_qubits, circuit_id = job_data.circuit_identifiers[curr_idx]
        result_object = CountsWrapper(counts)

        # Issue (#731) in QC-App-Oriented-Benchmarks will clean this if/else block.
        if JobType(benchmark_name) == JobType.PHASE_ESTIMATION:
            # Requires slightly different arguments.
            _, fidelity = benchmark.analyze_and_print_result(
                None, result_object, int(num_qubits) - 1, float(circuit_id), params["shots"]
            )
        elif JobType(benchmark_name) == JobType.QUANTUM_FOURIER_TRANSFORM:
            # Requires slightly different arguments.
            _, fidelity = benchmark.analyze_and_print_result(
                None,
                result_object,
                int(num_qubits),
                int(circuit_id),
                params["shots"],
                params["method"],
            )
        else:
            # Default call for Bernstein-Vazirani and Hidden Shift.
            _, fidelity = benchmark.analyze_and_print_result(
                None, result_object, int(num_qubits), int(circuit_id), params["shots"]
            )

        metrics.store_metric(num_qubits, circuit_id, "fidelity", fidelity)

    return metrics.circuit_metrics


def get_circuits_and_metrics(
    benchmark_name: str,
    params: dict[str, float | str],
) -> tuple[list[QuantumCircuit], QEDC_Metrics, list[tuple[str, str]]]:
    """
    Uses QED-C submodule to obtain circuits and circuit metrics.

    Args:
        params: the parameters to run the benchmark with, also includes benchmark_name.

    Returns:
        circuits: the list of quantum circuits for the benchmark.
        circuit_metrics: the circuit metrics at the time of circuit creation.
        circuit_identifiers: the unique identifiers for each circuit (num qubits, circuit id).
    """
    # Import the correct module
    benchmark = import_benchmark_module(benchmark_name)

    # Call the QED-C submodule to get the circuits and creation information.
    # Conver to QED-C parameter naming conventions.
    qedc_params = params.copy()
    if "shots" in qedc_params:
        qedc_params["num_shots"] = qedc_params.pop("shots")
    circuits, circuit_metrics = benchmark.run(
        **qedc_params,
        api="qiskit",
        get_circuits=True,
    )

    # Remove the subtitle key to keep our desired format.
    circuit_metrics.pop("subtitle", None)

    # Store the circuit identifiers and a flat list of circuits.
    circuit_identifiers = []
    flat_circuits = []
    for num_qubits in circuit_metrics.keys():
        for circuit_id in circuit_metrics[num_qubits].keys():
            circuit_identifiers.append((num_qubits, circuit_id))
            flat_circuits.append(circuits[num_qubits][circuit_id])

    return flat_circuits, circuit_metrics, circuit_identifiers


class QEDCBenchmark(Benchmark):
    """Benchmark class for QED-C experiments."""

    def _build_circuits(
        self, device: "QuantumDevice"
    ) -> tuple[list[QuantumCircuit], QEDC_Metrics, list[tuple[str, str]]]:
        """Shared circuit construction logic.

        Args:
            device: The quantum device to build circuits for.

        Returns:
            Tuple of (circuits, circuit_metrics, circuit_identifiers).
        """
        benchmark_name = self.params.benchmark_name
        circuits, circuit_metrics, circuit_identifiers = get_circuits_and_metrics(
            benchmark_name=benchmark_name,
            params=self.params.model_dump(exclude={"benchmark_name"}),
        )
        return circuits, circuit_metrics, circuit_identifiers

    def dispatch_handler(self, device: "QuantumDevice") -> QEDCData:
        # For more information on the parameters, view the schema for this benchmark.
        circuits, circuit_metrics, circuit_identifiers = self._build_circuits(device)

        return QEDCData.from_quantum_job(
            quantum_job=device.run(circuits, shots=self.params.shots),
            circuit_metrics=circuit_metrics,
            circuit_identifiers=circuit_identifiers,
        )

    def poll_handler(
        self,
        job_data: QEDCData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> QEDCResult:
        counts_list = flatten_counts(result_data)

        # Call the QED-C method after some pre-processing to obtain metrics.
        circuit_metrics = analyze_results(self.params.model_dump(), job_data, counts_list)

        return QEDCResult(circuit_metrics=circuit_metrics)

    def estimate_resources_handler(
        self,
        device: "QuantumDevice",
    ) -> list[CircuitBatch]:
        circuits, _, _ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
