from qedc._common import metrics
from qbraid.runtime.result_data import MeasCount
import importlib
from qiskit import QuantumCircuit
from enum import StrEnum
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metriq_gym.benchmarks.qedc_benchmarks import QEDCData


class QEDC_Benchmark_Names(StrEnum):
    """Store names of all supported QEDC Benchmarks"""

    BERNSTEIN_VAZIRANI = "Bernstein-Vazirani"
    PHASE_ESTIMATION = "Phase Estimation"
    HIDDEN_SHIFT = "Hidden Shift"
    QUANTUM_FOURIER_TRANSFORM = "Quantum Fourier Transform"


QEDC_BENCHMARK_IMPORTS: dict[QEDC_Benchmark_Names, str] = {
    QEDC_Benchmark_Names.BERNSTEIN_VAZIRANI: "qedc.bernstein_vazirani.bv_benchmark",
    QEDC_Benchmark_Names.PHASE_ESTIMATION: "qedc.phase_estimation.pe_benchmark",
    QEDC_Benchmark_Names.HIDDEN_SHIFT: "qedc.hidden_shift.hs_benchmark",
    QEDC_Benchmark_Names.QUANTUM_FOURIER_TRANSFORM: "qedc.quantum_fourier_transform.qft_benchmark",
}

"""
Type: QEDC_Metrics
Description: 
    The structure for all returned QEDC circuit metrics. 
    The first key represents the number of qubits for the group of circuits.
    The second key represents the unique identifier (secret str) for a circuit. 
    The third key represents the metric being stored.
Example:
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


def import_benchmark_module(benchmark_name: str) -> types.ModuleType:
    """
    Import the correct module

    Args:
        benchmark_name: the name of the benchmark being ran.

    Returns:
        benchmark: the module imported for the benchmark.

    """

    module_name = QEDC_BENCHMARK_IMPORTS[QEDC_Benchmark_Names(benchmark_name)]

    return importlib.import_module(module_name)


def analyze_results(
    params: dict[str, float | str], job_data: "QEDCData", counts_list: list[MeasCount]
) -> QEDC_Metrics:
    """
    Iterates over each circuit group and secret int to process results.
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

    # Initialize metrics module using qiskit in QED-C submodule.
    benchmark.qedc_benchmarks_init(api="qiskit")

    # Restore circuit metrics dictionary from the dispatch data
    metrics.circuit_metrics = job_data.circuit_metrics

    # Iterate and get the metrics for each circuit in the list.
    for curr_idx, (num_qubits, s_str) in enumerate(job_data.circuit_identifiers):
        counts: dict[str, int] = counts_list[curr_idx]

        result_object = CountsWrapper(counts)

        if benchmark_name == QEDC_Benchmark_Names.PHASE_ESTIMATION:
            # Requires slightly different arguments.
            _, fidelity = benchmark.analyze_and_print_result(
                None, result_object, int(num_qubits) - 1, float(s_str), params["num_shots"]
            )

        elif benchmark_name == QEDC_Benchmark_Names.QUANTUM_FOURIER_TRANSFORM:
            # Requires slightly different arguments.
            _, fidelity = benchmark.analyze_and_print_result(
                None,
                result_object,
                int(num_qubits),
                int(s_str),
                params["num_shots"],
                params["method"],
            )

        else:
            # Default call for Bernstein-Vazirani and Hidden Shift.
            _, fidelity = benchmark.analyze_and_print_result(
                None, result_object, int(num_qubits), int(s_str), params["num_shots"]
            )

        metrics.store_metric(num_qubits, s_str, "fidelity", fidelity)

    return metrics.circuit_metrics


def get_circuits_and_metrics(
    params: dict[str, float | str],
) -> tuple[list[QuantumCircuit], QEDC_Metrics, list[tuple[str, str]]]:
    """
    Uses QED-C submodule to obtain circuits and circuit metrics.

    Args:
        params: the parameters to run the benchmark with, also includes benchmark_name.

    Returns:
        circuits: the list of quantum circuits for the benchmark.
        circuit_metrics: the circuit metrics at the time of circuit creation.
        circuit_identifiers: the unique identifiers for each circuit (num qubits, secret str).
    """

    # Import the correct module
    benchmark = import_benchmark_module(str(params["benchmark_name"]))

    # Remove benchmark_name so it can be passed into the run call
    params.pop("benchmark_name", None)

    # Call the QED-C submodule to get the circuits and creation information.
    circuits, circuit_metrics = benchmark.run(
        **params,
        get_circuits=True,
    )

    # Remove the subtitle key to keep our desired format.
    circuit_metrics.pop("subtitle", None)

    # Store the circuit identifiers and a flat list of circuits.
    circuit_identifiers = []
    flat_circuits = []
    for num_qubits in circuit_metrics.keys():
        for s_str in circuit_metrics[num_qubits].keys():
            circuit_identifiers.append((num_qubits, s_str))
            flat_circuits.append(circuits[num_qubits][s_str])

    return flat_circuits, circuit_metrics, circuit_identifiers
