"""QEM-aware poll logic."""

from typing import Any, TYPE_CHECKING

from metriq_gym.qem.registry import build_pipeline
from metriq_gym.qem.technique import MitigationMetadata
from metriq_gym.helpers.task_helpers import flatten_counts

if TYPE_CHECKING:
    from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData, BenchmarkResult
    from qbraid import GateModelResultData, QuantumJob


def poll_with_qem(
    handler: "Benchmark",
    job_data: "BenchmarkData",
    result_data: list["GateModelResultData"],
    quantum_jobs: list["QuantumJob"],
    qem_data: dict[str, Any],
) -> tuple["BenchmarkResult", "BenchmarkResult"]:
    """Orchestrate QEM-aware polling, producing both mitigated and raw results.

    1. Flatten raw measurement counts from result_data.
    2. Reconstruct MitigationPipeline from stored config.
    3. Apply postprocess_counts to produce mitigated counts.
    4. Extract the unscaled (scale=1.0) raw counts for comparison.
    5. Call poll_handler with both mitigated and raw counts.

    Args:
        handler: The benchmark handler instance.
        job_data: The benchmark's intermediate data.
        result_data: Raw GateModelResultData from the provider.
        quantum_jobs: The quantum job objects.
        qem_data: Serialized QEM metadata from dispatch.

    Returns:
        Tuple of (mitigated_result, raw_result).
    """
    from qbraid.runtime.result_data import GateModelResultData as GMRD

    pipeline = build_pipeline(qem_data["config"])
    metadata_list = [
        MitigationMetadata(
            technique_name=m["technique_name"],
            data=m["data"],
        )
        for m in qem_data["metadata"]
    ]

    # Flatten all counts from provider results
    all_counts = flatten_counts(result_data)

    # Post-process to get mitigated counts (N results from N*K expanded)
    mitigated_counts = pipeline.postprocess_counts(all_counts, metadata_list)

    # Extract raw (unscaled) counts for comparison
    raw_counts = _extract_unscaled_counts(all_counts, qem_data)

    # Repackage counts into GateModelResultData for poll_handler
    mitigated_result_data = [GMRD(measurement_counts=c) for c in mitigated_counts]
    raw_result_data = [GMRD(measurement_counts=c) for c in raw_counts]

    mitigated_result = handler.poll_handler(job_data, mitigated_result_data, quantum_jobs)
    raw_result = handler.poll_handler(job_data, raw_result_data, quantum_jobs)

    return mitigated_result, raw_result


def _extract_unscaled_counts(
    all_counts: list[dict[str, int]],
    qem_data: dict[str, Any],
) -> list[dict[str, int]]:
    """Extract the unscaled (scale_factor=1.0) counts from the expanded results.

    For ZNE with N original circuits and K scale factors, all_counts has N*K entries.
    This extracts the N entries corresponding to scale_factor=1.0.
    """
    original_count = qem_data["original_circuit_count"]

    # Walk through metadata to find the total expansion and unscaled indices.
    # For a single ZNE technique, the pattern is interleaved:
    # [c0_s0, c0_s1, ..., c0_sK, c1_s0, ...]
    # We need the entry at the index of scale_factor=1.0 for each original circuit.
    metadata_list = qem_data["metadata"]
    if not metadata_list:
        return all_counts

    # For simplicity, handle the common case: single technique with scale_factors
    # For composed pipelines, the first technique's metadata determines the structure
    first_meta = metadata_list[0]
    scale_factors = first_meta["data"].get("scale_factors")
    if scale_factors is None:
        # Technique doesn't expand circuits (e.g., DD) — all counts are "raw"
        return all_counts

    num_scales = len(scale_factors)
    try:
        unscaled_idx = scale_factors.index(1.0)
    except ValueError:
        # No unscaled entry — just return the first scale of each circuit
        unscaled_idx = 0

    raw_counts = []
    for i in range(original_count):
        raw_counts.append(all_counts[i * num_scales + unscaled_idx])

    return raw_counts
