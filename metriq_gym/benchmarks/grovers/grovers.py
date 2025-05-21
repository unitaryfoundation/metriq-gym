"""
Grover's Benchmark for metriq-gym
Credit to QED-C for implementing the benchmark.
Reference: https://github.com/SRI-International/QC-App-Oriented-Benchmarks/tree/master

A rough draft for implementing QED-C's Grover's Benchmark into metriq-gym.
The benchmark generates N circuits for X qubits ranging from min_qubits to max_qubits.
Each circuit is then ran, and the polorization and Hellinger fidelities are calculated.
"""

from dataclasses import dataclass

import numpy as np
from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount

from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts

from metriq_gym.benchmarks.grovers.grovers_kernel import GroversSearch
from metriq_gym.benchmarks.grovers.metrics import polarization_fidelity


@dataclass
class GroversResult(BenchmarkResult):
    """Stores the results from running Grovers Benchmark.
    Results:
        fidelities: the polorization and Hellinger fidelity for each circuit in each number of qubits;
                    a fidelity dictionary has the following keys: "fidelity" and "hf_fidelity".
        all_num_qubits: the range of qubits used to generate circuits.
        marked_items: a list of secret strings used for each circuit in each number of qubits.

    Example results:
        Assume fidelities = [[d_1, d_2, d_3], [d_4, d_5, d_6]]
            where d_x represents a fidelity dictionary for a circuit x.
        Then say all_num_qubits = [2, 3] and marked_items = [[1, 2, 3], [1, 2, 4]].
        Then [d_1, d_2, d_3] are all circuits with 2 qubits and each circuit
        corresponds to a secret string as follows: d_1 = 1, d_2 = 2, d_3 = 3.
    """

    fidelities: list[list[dict[str, float]]]
    all_num_qubits: list[int]
    marked_items: list[list[int]]


@dataclass
class GroversData(BenchmarkData):
    """Stores the input parameters or metadata for Grovers Benchmark.
    Paramters/Metadata:
        shots: number of shots for each grover circuit to be ran with.
        min_qubits: minimum number of qubits to start generating circuits for the benchmark.
        max_qubits: maximum number of qubits to stop generating circuits for the benchmark.
        skip_qubits: the step size for generating circuits from the min to max qubit sizes.
        max_circuits: maximum number of circuits generated for each qubit size in the benchmark.
        marked_items: a list of secret strings used for each circuit in each number of qubits.
        all_num_qubits: the range of qubits used to generate circuits.
        use_mcx_shim: toggles a custom implementation of mcx gates.
    """

    shots: int
    min_qubits: int
    max_qubits: int
    skip_qubits: int
    max_circuits: int
    marked_items: list[list[int]]
    all_num_qubits: list[int]
    use_mcx_shim: bool


def create_circuits(
    min_qubits: int, max_qubits: int, skip_qubits: int, max_circuits: int, use_mcx_shim: bool
) -> tuple[list[QuantumCircuit], list[list[int]], list[int]]:
    """
    Modified version of the run() function in QED-C's Grover's Benchmark implementation.
    Reference: https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/grovers/qiskit/grovers_benchmark.py
    Args:
        min_qubits: minimum number of qubits to start generating circuits for the benchmark.
        max_qubits: maximum number of qubits to stop generating circuits for the benchmark.
        skip_qubits: the step size for generating circuits from the min to max qubit sizes.
        max_circuits: maximum number of circuits generated for each qubit size in the benchmark.
        use_mcx_shim: toggles a custom implementation of mcx gates.
    Returns:
        1. A list of quantum circuits to run for the benchmark
        2. A 2-D list where the first dimension consists of marked items and the second dimension
            corresponds to the number of qubits the circuit had.
                i.e.: [[1, 2, 3], [4, 5, 6]]
                    Would have secret strings [1, 2, 3] for X qubits
                    and secret strings [4, 5, 6] for Y qubits.
        3. A list of all qubit sizes used to generate circuits.
    Notes:
        The range from min to max qubits is inclusive.
    """

    # Create Grovers Circuits
    grovers_circuits = []
    all_num_qubits = list(range(min_qubits, max_qubits + 1, skip_qubits))
    secrets = []
    for num_qubits in all_num_qubits:
        # determine number of circuits to execute for this group
        num_circuits = min(2 ** (num_qubits), max_circuits)

        # print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # determine range of secret strings to loop over
        if 2 ** (num_qubits) <= max_circuits:
            s_range = list(range(num_circuits))
        else:
            # create selection larger than needed and remove duplicates (faster than random.choice())
            s_range = np.random.randint(1, 2 ** (num_qubits), num_circuits + 10)
            s_range = list(set(s_range))[0:max_circuits]

        # loop over limited # of secret strings for this
        for s_int in s_range:
            # create the circuit for given qubit size and secret string

            n_iterations = int(np.pi * np.sqrt(2**num_qubits) / 4)

            qc = GroversSearch(num_qubits, s_int, n_iterations, use_mcx_shim)

            grovers_circuits.append(qc)

        # Store the secret strings for later processing
        secrets.append(s_range)

    return (grovers_circuits, secrets, all_num_qubits)


def calc_fidelities(data: GroversData, counts: list[MeasCount]) -> list[list[dict[str, float]]]:
    """
    Calculates the fidelity for each job.
    Args:
        data: contains dispatch data like marked_items, all_num_qubits, and so on.
        counts: contains results from the quantum device (one MeasCount per circuit).
    Returns:
        A 2-D list of fidelities. Where the first dimension has the fidelity dictionaries
        and the second dimension corresponds to the number of qubits for the group.
    Notes:
        - The order of the counts is preserved, meaning that the index of each MeasCount
          corresponds to the index of the dispatch data for it.
            i.e.: counts[0] will represent the circuit with qubits equal to
                  all_num_qubits[0] and secret string of marked_items[0][0].
    """

    def analyze_results(
        counts: dict[str, int], num_qubits: int, marked_item: int
    ) -> dict[str, float]:
        """
        Computes the fidelity for one circuit.
        Slightly modified from QED-C's implementation in Grover's Benchmark.
        Args:
            counts: A dictionary of bitstrings to counts measured from the backend.
            num_qubits: How many qubits the circuit had.
            marked_item: The secret string for the circuit.
        Returns:
            Polarization fidelity and the hellinger fidelity as a dict with the following keys:
                fidelity
                hf_fidelity
        """
        correct_dist = grovers_dist(num_qubits, marked_item)
        fidelity = polarization_fidelity(counts, correct_dist)
        return fidelity

    def grovers_dist(num_qubits, marked_item):
        """
        Helper function for anaylze_results().
        Taken from QED-C's implementation of Grover's Benchmark.
        """
        n_iterations = int(np.pi * np.sqrt(2**num_qubits) / 4)

        dist = {}

        for i in range(2**num_qubits):
            key = bin(i)[2:].zfill(num_qubits)
            theta = np.arcsin(1 / np.sqrt(2**num_qubits))

            if i == int(marked_item):
                dist[key] = np.sin((2 * n_iterations + 1) * theta) ** 2
            else:
                dist[key] = (
                    np.cos((2 * n_iterations + 1) * theta) / (np.sqrt(2**num_qubits - 1))
                ) ** 2
        return dist

    # Iterate over each group of X qubits.
    # Compute the fidelity for each circuit run in the group.
    fidelities = []
    counts_idx = 0
    all_num_qubits = data.all_num_qubits
    for i in range(len(all_num_qubits)):
        num_qubits = all_num_qubits[i]
        marked_items_group = data.marked_items[i]

        current_group_fidelities = []

        for j in range(len(marked_items_group)):
            current_group_fidelities.append(
                analyze_results(
                    counts=counts[counts_idx],
                    num_qubits=num_qubits,
                    marked_item=marked_items_group[j],
                )
            )
            counts_idx += 1

        fidelities.append(current_group_fidelities)

    return fidelities


class Grovers(Benchmark):
    """Benchmark class for Grovers experiments."""

    def dispatch_handler(self, device: QuantumDevice) -> GroversData:
        # For more information on the parameters, view the schema for this benchmark.
        shots = self.params.shots
        min_qubits = self.params.min_qubits
        max_qubits = self.params.max_qubits
        skip_qubits = self.params.skip_qubits
        max_circuits = self.params.max_circuits
        use_mcx_shim = self.params.use_mcx_shim

        grovers_circuits, marked_items, all_num_qubits = create_circuits(
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            skip_qubits=skip_qubits,
            max_circuits=max_circuits,
            use_mcx_shim=use_mcx_shim,
        )

        quantum_job: QuantumJob | list[QuantumJob] = device.run(grovers_circuits, shots=shots)
        provider_job_ids = (
            [quantum_job.id]
            if isinstance(quantum_job, QuantumJob)
            else [job.id for job in quantum_job]
        )

        return GroversData(
            provider_job_ids=provider_job_ids,
            shots=shots,
            min_qubits=min_qubits,
            max_qubits=max_qubits,
            skip_qubits=skip_qubits,
            max_circuits=max_circuits,
            marked_items=marked_items,
            all_num_qubits=all_num_qubits,
            use_mcx_shim=use_mcx_shim,
        )

    def poll_handler(
        self,
        job_data: GroversData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> GroversResult:
        return GroversResult(
            fidelities=calc_fidelities(job_data, flatten_counts(result_data)),
            all_num_qubits=job_data.all_num_qubits,
            marked_items=job_data.marked_items,
        )
