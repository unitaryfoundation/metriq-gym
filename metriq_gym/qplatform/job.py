
"""Job introspection and execution utilities for quantum computing platforms.

This module provides standardized access to job execution details and metadata,
supporting both remote jobs (via qBraid) and local synchronous jobs.
"""

from functools import singledispatch
import json
import os
import tempfile
from datetime import datetime
from typing import Any

from qbraid import QuantumJob
from qbraid.runtime import QiskitJob, AzureQuantumJob, BraketQuantumTask
from qbraid.runtime.result_data import GateModelResultData, MeasCount
from qiskit_ibm_runtime.execution_span import ExecutionSpans


class LocalJob:
    """Local job wrapper for synchronous simulator execution.
    
    This class provides a qBraid-compatible interface for local simulator jobs.
    It is designed to be adapter-agnostic for better modularity.
    """
    
    def __init__(self, job_id: str, measurement_counts: list[MeasCount], cache_dir: str):
        """Initialize a local job with processed results.
        
        Args:
            job_id: Unique identifier for this job
            measurement_counts: Standardized measurement counts from simulator
            cache_dir: Directory for caching job data
        """
        self.id = job_id
        self._measurement_counts = measurement_counts
        self._cache_dir = cache_dir
        self._start_time = datetime.now()
        self._end_time = datetime.now()  # Immediate completion for local jobs
        
        # Cache results to disk for poll operations
        self._cache_results()
    
    def _cache_results(self):
        """Cache job results to disk for later retrieval."""
        cache_file = os.path.join(self._cache_dir, f"{self.id}.json")
        
        # Convert measurement counts to serializable format
        serialized_results = []
        for i, counts in enumerate(self._measurement_counts):
            serialized_results.append({
                'type': 'counts',
                'counts': dict(counts),
                'index': i
            })
        
        job_data = {
            'job_id': self.id,
            'results': serialized_results,
            'start_time': self._start_time.isoformat(),
            'end_time': self._end_time.isoformat(),
            'status': 'COMPLETED',
        }
        
        with open(cache_file, 'w') as f:
            json.dump(job_data, f, indent=2)
    
    def status(self):
        """Get job status.
        
        Returns:
            str: Always 'COMPLETED' for local jobs since they execute immediately
        """
        from qbraid.runtime import JobStatus
        return JobStatus.COMPLETED
    
    def result(self):
        """Get job results.
        
        Returns:
            LocalJobResult: Result object compatible with qBraid interface
        """
        return LocalJobResult(self._measurement_counts)
    
    def metadata(self):
        """Get job metadata.
        
        Returns:
            dict: Job metadata including timing and status information
        """
        return {
            'job_id': self.id,
            'status': 'COMPLETED',
            'start_time': self._start_time.isoformat(),
            'end_time': self._end_time.isoformat(),
            'execution_time': (self._end_time - self._start_time).total_seconds(),
            'local': True
        }


class LocalJobResult:
    """Result wrapper for local job execution.
    
    This class provides a qBraid-compatible interface for local simulator results.
    """
    
    def __init__(self, measurement_counts: list[MeasCount]):
        """Initialize result wrapper.
        
        Args:
            measurement_counts: Standardized measurement counts
        """
        self._measurement_counts = measurement_counts
        self.data = self._convert_to_gate_model_result()
    
    def _convert_to_gate_model_result(self) -> GateModelResultData:
        """Convert measurement counts to GateModelResultData format.
        
        Returns:
            GateModelResultData: Standardized result data
        """
        # Return based on structure expected by benchmarks
        if len(self._measurement_counts) == 1:
            return GateModelResultData(measurement_counts=self._measurement_counts[0])
        else:
            return GateModelResultData(measurement_counts=self._measurement_counts)


def load_local_job(job_id: str, **kwargs) -> LocalJob:
    """Load a local job from cached results.
    
    Args:
        job_id: The job identifier to load
        **kwargs: Additional arguments (for compatibility with load_job interface)
        
    Returns:
        LocalJob: The loaded local job
        
    Raises:
        FileNotFoundError: If job cache file is not found
        ValueError: If job data is corrupted
    """
    # Find the cache file - check common temp directories
    cache_file = None
    temp_dirs = [tempfile.gettempdir()]
    
    # Look for cache directories with metriq_local_ prefix
    for temp_dir in temp_dirs:
        if not os.path.exists(temp_dir):
            continue
        for item in os.listdir(temp_dir):
            if item.startswith("metriq_local_"):
                potential_cache = os.path.join(temp_dir, item, f"{job_id}.json")
                if os.path.exists(potential_cache):
                    cache_file = potential_cache
                    break
        if cache_file:
            break
    
    if not cache_file:
        raise FileNotFoundError(f"Local job cache not found for job {job_id}")
    
    try:
        with open(cache_file, 'r') as f:
            job_data = json.load(f)
        
        # Reconstruct measurement counts from cached data
        measurement_counts = []
        for result_data in job_data['results']:
            if result_data['type'] == 'counts':
                measurement_counts.append(MeasCount(result_data['counts']))
        
        cache_dir = os.path.dirname(cache_file)
        return LocalJob(job_id, measurement_counts, cache_dir)
        
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Corrupted job cache for job {job_id}: {e}")


@singledispatch
def execution_time(quantum_job: QuantumJob) -> float:
    raise NotImplementedError(f"Execution time not implemented for type {type(quantum_job)}")


@execution_time.register
def _(quantum_job: QiskitJob) -> float:
    execution_spans: ExecutionSpans = quantum_job._job.result().metadata["execution"][
        "execution_spans"
    ]
    return (execution_spans.stop - execution_spans.start).total_seconds()


@execution_time.register
def _(quantum_job: AzureQuantumJob) -> float:
    start_time = quantum_job._job.details.begin_execution_time
    end_time = quantum_job._job.details.end_execution_time
    if start_time is None or end_time is None:
        raise ValueError("Execution time not available")
    return (end_time - start_time).total_seconds()


@execution_time.register
def _(quantum_job: BraketQuantumTask) -> float:
    # TODO: for speed benchmarking, we need 'execution' metadata instead of 'createdAt' and 'endedAt'
    return (
        quantum_job._task.metadata()["endedAt"] - quantum_job._task.metadata()["createdAt"]
    ).total_seconds()


@execution_time.register
def _(quantum_job: LocalJob) -> float:
    """Get execution time for local jobs.
    
    Args:
        quantum_job: LocalJob instance
        
    Returns:
        float: Execution time in seconds (typically very small for local jobs)
    """
    return (quantum_job._end_time - quantum_job._start_time).total_seconds()