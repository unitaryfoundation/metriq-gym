""" Grover's Benchmark for metriq-gym
Credit to QED-C for implementing the benchmark.

[Description]
""" 

from dataclasses import dataclass

import numpy as np
from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qbraid.runtime.result_data import MeasCount

from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
from metriq_gym.helpers.task_helpers import flatten_counts

from grovers_kernel import GroversSearch
from qedc_grovers_benchmark import grovers_dist
from metrics import polarization_fidelity

MAX_QUBITS=8
benchmark_name = "Grover's Search"
np.random.seed(0)
verbose = False

@dataclass
class GroversResult(BenchmarkResult):
    """Stores the result(s) from running Grovers Benchmark."""
    fidelities: list[dict[str, float]]
    all_num_qubits: list[int]
    marked_items: list[list[int]]

@dataclass
class GroversData(BenchmarkData):
    """Stores the input parameters or metadata for Grovers Benchmark."""
    min_qubits: int
    max_qubits: int
    skip_qubits: int
    max_circuits: int
    marked_items: list[list[int]]
    all_num_qubits: list[int]

def create_circuits(min_qubits=2, max_qubits=6, skip_qubits=1, max_circuits=3,
        use_mcx_shim=False) -> tuple[list[QuantumCircuit], list[list[int]], list[int]]:
    """
    Modified version of the run() function in grovers_benchmark.py. 
    Args:
        min_qubits: minimum number of qubits to start generating circuits for the benchmark. 
        max_qubits: maxiumum number of qubits to stop generating circuits for the benchmark.
        skip_qubits: the step size for generating circuits from the min to max qubit sizes. 
        max_circuits: maximum number of circuits generated for each qubit size in the benchmark.
        use_mcx_shim: for validating the implementation of an mcx shim.
    Returns:
        1. A list of quantum circuits to run for the benchmark
        2. A 2-D list where the first dimension consists of marked items and the second dimension
            corresponds to the number of qubits the circuit had. 
                i.e.: [[1, 2, 3], [4, 5, 6]] 
                    Would have secret strings [1, 2, 3] for X qubits
                    and secret strings [4, 5, 6] for Y qubits.
        3. A list of all qubit sizes used to generate circuits. 
    """
    # Clamp the maximum number of qubits
    if max_qubits > MAX_QUBITS:
        print(f"INFO: {benchmark_name} benchmark is limited to a maximum of {MAX_QUBITS} qubits.")
        max_qubits = MAX_QUBITS
    
    # validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    skip_qubits = max(1, skip_qubits)

    # Create Grovers Circuits
    grovers_circuits = []
    all_num_qubits = range(min_qubits, max_qubits + 1, skip_qubits)
    secrets = []
    for num_qubits in all_num_qubits:
        
        # determine number of circuits to execute for this group
        num_circuits = min(2 ** (num_qubits), max_circuits)

        #print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")
        
        # determine range of secret strings to loop over
        if 2**(num_qubits) <= max_circuits:
            s_range = list(range(num_circuits))
        else:
            # create selection larger than needed and remove duplicates (faster than random.choice())
            s_range = np.random.randint(1, 2**(num_qubits), num_circuits + 10)
            s_range = list(set(s_range))[0:max_circuits]
        
        # loop over limited # of secret strings for this
        for s_int in s_range:
            # create the circuit for given qubit size and secret string

            n_iterations = int(np.pi * np.sqrt(2 ** num_qubits) / 4)

            qc = GroversSearch(num_qubits, s_int, n_iterations, use_mcx_shim)

            grovers_circuits.append(qc)
        
        # Store the secret strings for later processing
        secrets.append(s_range)
    
    return (grovers_circuits, secrets, all_num_qubits)

def calc_fidelities(data: GroversData, counts: list[MeasCount]) -> list[dict[str, float]]:
    """
    Calculates the fidelity for each job.
    Args:
        data: contains dispatch data like marked_items, all_num_qubits, and so on. 
        counts: contains results from the quantum device (one MeasCount per circuit).
    Returns:
        A list of fidelities, each dictionary representing the fidelity of a run.
    Notes:
        - The order of the counts is preserved, meaning that the index of each MeasCount
          corresponds to the index of the dispatch data for it. 
    """
    def analyze_results(counts: dict[str, int], 
                        num_qubits: int, 
                        marked_item: int
                        ) -> dict[str, float]: 
        """
        Computes the fidelity for one job. 
        Args:
            counts: A dictionary of bitstrings to counts measured from the backend.
            num_qubits: How many qubits the job had. 
            marked_item: The secret of the  
        Returns:
            Polarization fidelity and the hellinger fidelity as a dict with the following keys:
                fidelity
                hf_fidelity
        """
        correct_dist = grovers_dist(num_qubits, marked_item)
        fidelity = polarization_fidelity(counts, correct_dist)
        return fidelity

    # Iterate over each group of X qubits.
    # Compute the fidelity for each run in the group. 
    fidelities = []
    counts_idx = 0
    all_num_qubits = data.all_num_qubits
    
    for i in range(len(all_num_qubits)):
        
        num_qubits = all_num_qubits[i]
        marked_items_group = data.marked_items[i]
        
        current_group_fidelities = []
        
        for j in range(len(marked_items_group)):
            
            current_group_fidelities.append(analyze_results(counts=counts[counts_idx],
                                                            num_qubits=num_qubits,
                                                            marked_item=marked_items_group[j]
                                                            )
                                            )
            counts_idx += 1
        
        fidelities.append(current_group_fidelities)
    
    return fidelities

class Grovers(Benchmark):
    """Benchmark class for Grovers experiments."""

    def dispatch_handler(
        self,
        device: QuantumDevice
    ) -> GroversData:
        
        # The parameters below can be modified to be added from the JSON file 
        # and accessed via self.params.X
        shots = self.params.shots or 100
        min_qubits = 2
        max_qubits = 6 
        skip_qubits = 1
        max_circuits = 3

        grovers_circuits, marked_items, all_num_qubits = create_circuits(min_qubits=min_qubits, 
                                                                        max_qubits=max_qubits,
                                                                        skip_qubits=skip_qubits, 
                                                                        max_circuits=max_circuits)

        quantum_job: QuantumJob | list[QuantumJob] = device.run(grovers_circuits, shots=shots)
        provider_job_ids = (
            [quantum_job.id]
            if isinstance(quantum_job, QuantumJob)
            else [job.id for job in quantum_job]
        )

        return GroversData(
            provider_job_ids = provider_job_ids,
            min_qubits = min_qubits,
            max_qubits = max_qubits,
            skip_qubits = skip_qubits,
            max_circuits = max_circuits,
            marked_items = marked_items,
            all_num_qubits = all_num_qubits
        )

    def poll_handler(
        self,
        job_data: GroversData,
        result_data: list[GateModelResultData]
    ) -> GroversResult:
        
        return GroversResult(
            fidelities = calc_fidelities(job_data, flatten_counts(result_data)),
            all_num_qubits = job_data.all_num_qubits,
            marked_items = job_data.marked_items
        )