"""QEM-aware dispatch logic."""

from typing import Any, TYPE_CHECKING

from metriq_gym.qem.registry import build_pipeline

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from metriq_gym.benchmarks.benchmark import Benchmark, BenchmarkData
    from qbraid import QuantumDevice


def _transpile_no_opt(
    circuits: list["QuantumCircuit"], device: "QuantumDevice"
) -> list["QuantumCircuit"]:
    """Transpile circuits to the device's basis gates without optimisation.

    Gate folding (used by ZNE) inserts G†G pairs that a normal transpiler
    would cancel out. Using ``optimization_level=0`` decomposes gates into the
    target basis so the backend can execute them, but does not simplify or
    cancel any gates, preserving the noise-amplification structure.
    """
    from qiskit import transpile

    # Use the underlying backend if available (LocalAerDevice stores it here).
    try:
        backend = device.profile.extra.get("backend")
    except (AttributeError, TypeError):
        return circuits

    # Check that the backend is a real Aer backend, not a mock
    try:
        from qiskit_aer import AerSimulator

        if not isinstance(backend, AerSimulator):
            return circuits
    except ImportError:
        return circuits

    return transpile(circuits, backend=backend, optimization_level=0)


def dispatch_with_qem(
    handler: "Benchmark",
    device: "QuantumDevice",
    qem_config: list[dict[str, Any]],
) -> tuple["BenchmarkData", dict[str, Any]]:
    """Orchestrate QEM-aware dispatch.

    1. Call handler.build_circuits(device) to get a CircuitPackage.
    2. Build the MitigationPipeline from config.
    3. Apply circuit transforms.
    4. Transpile to basis gates at optimization_level=0 (no gate cancellation).
    5. Submit transformed circuits to device.
    6. Build BenchmarkData via handler.create_job_data().
    7. Return both benchmark data and serialized QEM metadata.

    Args:
        handler: The benchmark handler instance.
        device: The quantum device to submit to.
        qem_config: List of technique config dicts from the benchmark schema.

    Returns:
        Tuple of (BenchmarkData, qem_data_dict) where qem_data_dict contains
        serialized metadata for persistence.
    """
    package = handler.build_circuits(device)
    pipeline = build_pipeline(qem_config)

    transformed_circuits, qem_metadata = pipeline.transform_circuits(
        package.circuits, package.shots
    )

    # Transpile to basis gates without optimisation so gate folds are preserved
    # but the backend can still execute the circuits.
    transformed_circuits = _transpile_no_opt(transformed_circuits, device)

    # Bypass qBraid's transpile/transform which would optimise away folded gates.
    saved_transpile = device._options.get("transpile")
    saved_transform = device._options.get("transform")
    try:
        device._options.transpile = False
        device._options.transform = False
        quantum_job = device.run(transformed_circuits, shots=package.shots)
    finally:
        device._options.transpile = saved_transpile
        device._options.transform = saved_transform

    # Build benchmark data using the original metadata + new job IDs
    job_data = handler.create_job_data(package, quantum_job)

    # Serialize QEM metadata for persistence in MetriqGymJob.data
    qem_data = {
        "config": qem_config,
        "metadata": [
            {"technique_name": m.technique_name, "data": m.data}
            for m in qem_metadata
        ],
        "original_circuit_count": len(package.circuits),
    }

    return job_data, qem_data
