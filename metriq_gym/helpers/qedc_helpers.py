from qedc._common import metrics
from qbraid.runtime.result_data import MeasCount
import importlib
from qiskit import QuantumCircuit
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metriq_gym.benchmarks.qedc_benchmarks import QEDCData

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
    """Import the correct module"""

    if benchmark_name.lower() == "bernstein-vazirani":
        module_name = "qedc.bernstein_vazirani.bv_benchmark"

    elif benchmark_name.lower() == "phase estimation":
        module_name = "qedc.phase_estimation.pe_benchmark"

    elif benchmark_name.lower() == "hidden shift":
        module_name = "qedc.hidden_shift.hs_benchmark"

    return importlib.import_module(module_name)


def analyze_results(job_data: "QEDCData", counts_list: list[MeasCount]) -> QEDC_Metrics:
    """
    Iterates over each circuit group and secret int to process results.
    Uses QED-C submodule to obtain calculations.

    Args:
        job_data: the BernsteinVaziraniData object for the job.
        counts_list: a list of all counts objects, each index corresponds to a circuit.

    Returns:
        circuit_metrics: the updated circuit metrics in QED-C's format.
    """

    class CountsWrapper:
        """
        A wrapper class to enable support with QED-C's method to analyze results.
        """

        def __init__(self, qc: QuantumCircuit, counts: dict[str, int]):
            self.qc = qc
            self.counts = counts

        def get_counts(self, qc):
            if qc == self.qc:
                return self.counts

    # Import the correct module
    benchmark = import_benchmark_module(job_data.benchmark_name)

    # Initialize metrics module in QED-C submodule.
    benchmark.qedc_benchmarks_init()

    # Restore circuit metrics dictionary from the dispatch data
    metrics.circuit_metrics = job_data.circuit_metrics

    # Iterate and get the metrics for each circuit in the list.
    for curr_idx, (num_qubits, s_str) in enumerate(job_data.circuit_identifiers):
        counts: dict[str, int] = counts_list[curr_idx]

        qc = job_data.circuits[curr_idx]

        result_object = CountsWrapper(qc, counts)

        if job_data.benchmark_name.lower() == "phase estimation":
            _, fidelity = benchmark.analyze_and_print_result(
                qc, result_object, int(num_qubits) - 1, float(s_str), job_data.shots
            )

        else:
            _, fidelity = benchmark.analyze_and_print_result(
                qc, result_object, int(num_qubits), int(s_str), job_data.shots
            )

        metrics.store_metric(num_qubits, s_str, "fidelity", fidelity)

    return metrics.circuit_metrics


def get_circuits_and_metrics(
    min_qubits: int,
    max_qubits: int,
    skip_qubits: int,
    max_circuits: int,
    shots: int,
    benchmark_name: str,
) -> tuple[list[QuantumCircuit], QEDC_Metrics, list[tuple[str, str]]]:
    # Import the correct module
    benchmark = import_benchmark_module(benchmark_name)

    # Call the QED-C submodule to get the circuits and creation information.
    # Note that phase estimation doesn't have a methods parameter.
    if benchmark_name.lower() != "phase estimation":
        circuits, circuit_metrics = benchmark.run(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            skip_qubits=skip_qubits,
            max_circuits=max_circuits,
            num_shots=shots,
            method=1,
            get_circuits=True,
        )
    else:
        circuits, circuit_metrics = benchmark.run(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            skip_qubits=skip_qubits,
            max_circuits=max_circuits,
            num_shots=shots,
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
