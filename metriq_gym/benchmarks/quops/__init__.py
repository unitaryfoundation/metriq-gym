r"""QUOPS (experimental).

Measures physical-qubit QUOPS using random Pauli rotations, random CNOT
pairings, and three-ensemble mirror circuit fidelity estimation (MCFE).
Width is the number of computational qubits; size counts one per rotation
and two per CNOT before compilation. Logical qubits and postselection are
outside this implementation.

Configuration and results:
    ``polarization_estimator="mcfe"`` and ``gate_set="pauli_rotations_cnot"``
    are the defaults and currently supported choices. The gate set specifies
    the paper's abstract workload; native gates are selected during compilation.
    Both choices are recorded in saved jobs and results.

    Declare a sequence of ``{width, size}`` shapes in strictly increasing size,
    within ``width**2 <= size <= width**3``. Each shape must consist of complete
    rounds plus an optional partial round. A seeded qubit subset is selected
    once per width and reused for its circuits and SPAM references.

    ``quops`` is a lower bound from the predeclared sequence: each one-sided
    95% polarization bound must reach ``exp(-0.5)`` before advancing. The first
    failed or unresolved test ends certification; subsequent points remain in
    ``shapes`` as diagnostics. ``scan_status=limit_reached`` means all tested
    shapes passed, so extend the scan in a separately planned experiment.
    Bootstrap uncertainty covers sampling, not the approximate MCFE model.

    ``quops_per_second = 2 * size * polarization**2 * mirror_shots / seconds``
    uses measured QPU time for every mirror job of the winning shape, excluding
    SPAM jobs. Provider adapters supply seconds and their source through
    ``qplatform.job.qpu_time``; IBM currently supplies finalized QPU usage.
    The rate is null when suitable timing is unavailable, including local
    simulation or pending accounting. Poll again with ``--no-cache`` once
    accounting is complete. Measured times and sources are saved for replay.
    Raw counts, ideal bitstrings, layouts, compilation metadata, circuit hashes
    and seeds support replay.

Resource estimates:
    Estimates sample up to five workloads per shape and ten SPAM circuits per
    width, scaling their gate counts and timing to the full workload. Circuit,
    shot and batch totals are exact. Add ``--full`` to estimate every circuit;
    this can take several minutes. Dispatch always builds every configured circuit.

Published comparison:
    The published Boston result is QUOPS 204 at width 6, approximately 310,000
    QUOPS/s. Our regression test recovers this from the authors' summary data;
    it is not a new hardware measurement. A fresh run may differ with calibration,
    layout, compiler version and batching. Here each shape is timed separately;
    the paper approximates rates from mixed-shape batches. A scan restricted
    to one width does not establish a maximum across every width.

Providers:
    IBM uses Qiskit's default layout selection with backend calibration data
    at optimization level 3 before mirroring. The selected subset is reused
    for every circuit at that width; ``physical_qubits`` can fix it explicitly.
    Local Aer also uses Qiskit compilation. Quantinuum H-series uses the same
    sampled circuits with native lowering and preserved barriers, without
    per-shot randomization; Helios needs a future runtime adapter. Braket
    hardware requires native Rx/Rz/CZ or Rx/Rz/CNOT gates and verbatim execution;
    unsupported devices fail before submission.
    Provider-specific timing must be validated before enabling their rates.
    ``quops.example.json`` is a small local smoke test, not a hardware preset.

References:
    - [QUOPS paper, especially Supplementary Sections II, IV and IX](https://arxiv.org/abs/2609.12146).
    - [Authors' implementations and results](https://github.com/sandialabs/PRAQTICE/tree/quops/quops).
"""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from typing import Any, TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from qiskit import QuantumCircuit, qasm2

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.qplatform.device import validate_qubit_capacity
from metriq_gym.qplatform.job import QPUTime, qpu_time, sum_qpu_times
from metriq_gym.qplatform.quops import (
    compile_quops_circuit,
    prepare_quops_circuit,
    submit_quops_circuits,
    validate_quops_options,
)
from metriq_gym.resource_estimation import CircuitBatch, count_two_qubit_gates

from .analysis import QUOPS_THRESHOLD, ShapeEstimate, estimate_polarization, effective_polarization
from .analysis import gated_score, quops_rate
from .circuits import (
    build_mirror_circuits,
    build_spam_circuit,
    sample_quops_circuit,
    validate_shape,
)

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob

PROTOCOL = "physical-mcfe-v1"
DEFAULT_POLARIZATION_ESTIMATOR = "mcfe"
DEFAULT_GATE_SET = "pauli_rotations_cnot"


def _validate_protocol_options(polarization_estimator: str, gate_set: str) -> None:
    """Reject unimplemented choices before execution or saved-data analysis."""
    if polarization_estimator != DEFAULT_POLARIZATION_ESTIMATOR:
        raise ValueError(f"Unsupported QUOPS polarization_estimator: {polarization_estimator}")
    if gate_set != DEFAULT_GATE_SET:
        raise ValueError(f"Unsupported QUOPS gate_set: {gate_set}")


@dataclass
class QUOPSData(BenchmarkData):
    shapes: list[dict[str, int]]
    circuit_records: list[dict[str, Any]]
    batches: list[dict[str, Any]]
    compilation: dict[str, Any]
    seed: int
    shots: int
    bootstrap_samples: int
    protocol: str = PROTOCOL
    polarization_estimator: str = DEFAULT_POLARIZATION_ESTIMATOR
    gate_set: str = DEFAULT_GATE_SET


class QUOPSShapeResult(BaseModel):
    width: int
    size: int
    polarization: BenchmarkScore | None
    lower_bound: float | None
    passed: bool
    certified: bool
    ensemble_means: dict[str, float]
    quops_per_second: float | None = None
    execution_seconds: float | None = None
    timing_source: str | None = None


class QUOPSResult(BenchmarkResult):
    quops: int
    width: int | None
    quops_per_second: float | None
    scan_status: str
    shapes: list[QUOPSShapeResult]
    protocol: str = PROTOCOL
    polarization_estimator: str = DEFAULT_POLARIZATION_ESTIMATOR
    gate_set: str = DEFAULT_GATE_SET

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=self.quops)


class QUOPS(Benchmark[QUOPSData, QUOPSResult]):
    def _configuration(self) -> dict[str, Any]:
        config = self.params.model_dump()
        _validate_protocol_options(config["polarization_estimator"], config["gate_set"])
        shapes = config["shapes"]
        previous_size = 0
        for shape in shapes:
            width, size = shape["width"], shape["size"]
            validate_shape(width, size)
            if not width**2 <= size <= width**3:
                raise ValueError("QUOPS shapes must satisfy width^2 <= size <= width^3")
            if size <= previous_size:
                raise ValueError("QUOPS shapes must have strictly increasing sizes")
            previous_size = size
        layout = config["physical_qubits"]
        if layout is not None and any(shape["width"] != len(layout) for shape in shapes):
            raise ValueError("physical_qubits must match the width of every configured shape")
        return config

    def _build(self, device: "QuantumDevice") -> tuple[list[CircuitBatch], QUOPSData]:
        config = self._configuration()
        if device is None:
            raise ValueError("QUOPS estimation requires --device for compilation and layout")
        validate_quops_options(device, dynamical_decoupling=config["dynamical_decoupling"])
        validate_qubit_capacity(device, max(shape["width"] for shape in config["shapes"]))
        rng = np.random.default_rng(config["seed"])
        circuits: list[QuantumCircuit] = []
        records: list[dict[str, Any]] = []
        input_counts: list[int] = []
        batches: list[CircuitBatch] = []
        batch_records: list[dict[str, Any]] = []
        compilation: dict[str, Any] = {
            "qiskit_version": version("qiskit"),
            "qbraid_version": version("qbraid"),
            "dynamical_decoupling": config["dynamical_decoupling"],
            "layouts": {},
        }
        layouts = {}
        for width in sorted({shape["width"] for shape in config["shapes"]}):
            if getattr(self.args, "verbose", False):
                print(f"QUOPS: selecting a qubit subset for width {width}...")
            size = max(shape["size"] for shape in config["shapes"] if shape["width"] == width)
            placement = compile_quops_circuit(
                device,
                sample_quops_circuit(width, size, rng),
                seed=int(rng.integers(2**31)),
                physical_qubits=config["physical_qubits"],
            )
            layouts[width] = placement.physical_qubits
            compilation["layouts"][str(width)] = {
                "physical_qubits": placement.physical_qubits,
                **placement.metadata,
            }

        def append(mirror, width: int, shape_index: int | None, ensemble: str) -> None:
            input_counts.append(count_two_qubit_gates(mirror.circuit))
            circuit = prepare_quops_circuit(device, mirror.circuit, layouts[width])
            circuits.append(circuit)
            records.append(
                {
                    "shape_index": shape_index,
                    "width": width,
                    "ensemble": ensemble,
                    "ideal_bitstring": mirror.ideal_bitstring,
                    "circuit_sha256": sha256(qasm2.dumps(circuit).encode()).hexdigest(),
                }
            )

        def flush(shape_index: int | None) -> None:
            for start in range(0, len(circuits), config["max_circuits_per_job"]):
                batch = circuits[start : start + config["max_circuits_per_job"]]
                batches.append(CircuitBatch(circuits=batch, shots=config["shots"], transpile=False))
                batch_records.append(
                    {
                        "shape_index": shape_index,
                        "num_circuits": len(batch),
                        "provider_job_ids": [],
                        "execution_seconds": None,
                        "timing_source": None,
                    }
                )
            circuits.clear()

        for shape_index, shape in enumerate(config["shapes"]):
            width, size = shape["width"], shape["size"]
            if getattr(self.args, "verbose", False):
                print(
                    f"QUOPS: compiling {config['num_circuits']} workloads at ({width}, {size})..."
                )
            for _ in range(config["num_circuits"]):
                compiled = compile_quops_circuit(
                    device,
                    sample_quops_circuit(width, size, rng),
                    seed=int(rng.integers(2**31)),
                    physical_qubits=layouts[width],
                )
                mixed, reference = build_mirror_circuits(compiled.circuit, rng)
                append(mixed, width, shape_index, "mixed")
                append(reference, width, shape_index, "reference")
            flush(shape_index)
        for width in layouts:
            if getattr(self.args, "verbose", False):
                print(
                    f"QUOPS: preparing {config['num_spam_circuits']} SPAM circuits at width {width}..."
                )
            for _ in range(config["num_spam_circuits"]):
                append(build_spam_circuit(width, rng), width, None, "spam")
            flush(None)

        counts = [count_two_qubit_gates(circuit) for batch in batches for circuit in batch.circuits]
        return batches, QUOPSData(
            provider_job_ids=[],
            shapes=config["shapes"],
            circuit_records=records,
            batches=batch_records,
            compilation=compilation,
            seed=config["seed"],
            shots=config["shots"],
            bootstrap_samples=config["bootstrap_samples"],
            polarization_estimator=config["polarization_estimator"],
            gate_set=config["gate_set"],
            input_two_qubit_gate_counts=input_counts,
            transpiled_two_qubit_gate_counts=counts,
        )

    def dispatch_handler(self, device: "QuantumDevice") -> QUOPSData:
        batches, data = self._build(device)
        for batch, record in zip(batches, data.batches, strict=True):
            jobs = submit_quops_circuits(
                device,
                batch.circuits,
                shots=batch.shots,
                dynamical_decoupling=data.compilation["dynamical_decoupling"],
            )
            ids = [job.id for job in jobs]
            record["provider_job_ids"] = ids
            data.provider_job_ids.extend(ids)
        return data

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        config = self._configuration()
        num_circuits = config["num_circuits"]
        num_spam = config["num_spam_circuits"]
        if not getattr(self.args, "full", False):
            num_circuits = min(num_circuits, 5)
            num_spam = min(num_spam, 10)
        sampled = (num_circuits, num_spam) != (config["num_circuits"], config["num_spam_circuits"])
        # Copy parameters so estimation cannot change a later dispatch on this handler.
        params = self.params.model_copy(
            update={
                "num_circuits": num_circuits,
                "num_spam_circuits": num_spam,
            }
        )
        batches, data = QUOPS(self.args, params)._build(device)
        if sampled:
            groups: dict[tuple[int | None, int], list[QuantumCircuit]] = defaultdict(list)
            circuits = (circuit for batch in batches for circuit in batch.circuits)
            for circuit, record in zip(circuits, data.circuit_records, strict=True):
                groups[record["shape_index"], record["width"]].append(circuit)
            batches = []
            for (shape_index, _), sample in groups.items():
                count = (
                    config["num_spam_circuits"]
                    if shape_index is None
                    else 2 * config["num_circuits"]
                )
                # Repeat references, not copies: the IBM estimator schedules each
                # sampled circuit once while retaining actual shot/batch overheads.
                expanded = [sample[i % len(sample)] for i in range(count)]
                for start in range(0, count, config["max_circuits_per_job"]):
                    batches.append(
                        CircuitBatch(
                            circuits=expanded[start : start + config["max_circuits_per_job"]],
                            shots=config["shots"],
                            transpile=False,
                            sampled=True,
                        )
                    )
            batches[0].notes.append(
                f"QUOPS sampling: {num_circuits} workloads per shape and {num_spam} SPAM "
                "circuits per width. Gate counts and timing vary with the sampled circuits."
            )
        batches[0].notes.append(
            "QUOPS: covers mixed/reference mirrors and separate SPAM circuits at every "
            "configured shape. IBM dynamical-decoupling overhead is not estimated."
        )
        return batches

    def poll_handler(
        self,
        job_data: QUOPSData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> QUOPSResult:
        if job_data.protocol != PROTOCOL:
            raise ValueError(f"Unsupported QUOPS protocol: {job_data.protocol}")
        _validate_protocol_options(job_data.polarization_estimator, job_data.gate_set)
        counts = flatten_counts(result_data)
        if len(counts) != len(job_data.circuit_records):
            raise ValueError("QUOPS result count does not match the dispatched circuit count")
        values: dict[tuple[int | None, int, str], list[float]] = defaultdict(list)
        for count, record in zip(counts, job_data.circuit_records, strict=True):
            if sum(count.values()) != job_data.shots:
                raise ValueError("QUOPS requires the requested shot count for every circuit")
            key = (record["shape_index"], record["width"], record["ensemble"])
            values[key].append(effective_polarization(count, record["ideal_bitstring"]))

        jobs_by_id = {job.id: job for job in quantum_jobs}
        for batch in job_data.batches:
            # Earlier physical-mcfe-v1 exports stored only IBM usage seconds.
            # An explicitly null source in newer data must remain unresolved.
            if "timing_source" not in batch:
                batch["timing_source"] = (
                    "ibm_qpu_usage" if batch["execution_seconds"] is not None else None
                )
            if quantum_jobs and batch["shape_index"] is not None:
                jobs = [
                    jobs_by_id[job_id]
                    for job_id in batch["provider_job_ids"]
                    if job_id in jobs_by_id
                ]
                timing = (
                    sum_qpu_times(qpu_time(job) for job in jobs)
                    if len(jobs) == len(batch["provider_job_ids"])
                    else None
                )
                batch["execution_seconds"] = timing.seconds if timing is not None else None
                batch["timing_source"] = timing.source if timing is not None else None

        estimates = []
        rows = []
        gate_open = True
        for index, shape in enumerate(job_data.shapes):
            width, size = shape["width"], shape["size"]
            mixed = values[index, width, "mixed"]
            reference = values[index, width, "reference"]
            spam = values[None, width, "spam"]
            estimate = estimate_polarization(
                mixed,
                reference,
                spam,
                seed=job_data.seed + index,
                bootstrap_samples=job_data.bootstrap_samples,
            )
            estimates.append(ShapeEstimate(width, size, estimate))
            # A shape passes when its MCFE one-sided 95% lower bound reaches exp(-0.5).
            passed = estimate.lower_bound is not None and estimate.lower_bound >= QUOPS_THRESHOLD
            gate_open = gate_open and passed
            timing = sum_qpu_times(
                QPUTime(batch["execution_seconds"], batch["timing_source"])
                for batch in job_data.batches
                if batch["shape_index"] == index
            )
            seconds = timing.seconds if timing is not None else None
            rate = None
            if seconds is not None and estimate.value is not None and estimate.value >= 0:
                rate = quops_rate(
                    size, estimate.value, (len(mixed) + len(reference)) * job_data.shots, seconds
                )
            rows.append(
                QUOPSShapeResult(
                    width=width,
                    size=size,
                    polarization=(
                        BenchmarkScore(value=estimate.value, uncertainty=estimate.standard_error)
                        if estimate.value is not None
                        else None
                    ),
                    lower_bound=estimate.lower_bound,
                    passed=passed,
                    certified=gate_open,
                    ensemble_means={
                        "mixed": float(np.mean(mixed)),
                        "reference": float(np.mean(reference)),
                        "spam": float(np.mean(spam)),
                    },
                    quops_per_second=rate,
                    execution_seconds=seconds,
                    timing_source=timing.source if timing is not None else None,
                )
            )
        score, winning_shape = gated_score(estimates)
        winner = next((row for row in rows if (row.width, row.size) == winning_shape), None)
        status = "limit_reached" if gate_open else "threshold_reached"
        first_failure = next((row for row in rows if not row.passed), None)
        if first_failure is not None and first_failure.lower_bound is None:
            status = "unresolved"
        return QUOPSResult(
            quops=score,
            width=winner.width if winner else None,
            quops_per_second=winner.quops_per_second if winner else None,
            scan_status=status,
            shapes=rows,
            polarization_estimator=job_data.polarization_estimator,
            gate_set=job_data.gate_set,
        )
