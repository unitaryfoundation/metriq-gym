from argparse import Namespace
from dataclasses import asdict, dataclass
import json
from types import SimpleNamespace
from unittest.mock import Mock

from jsonschema import ValidationError
import pytest
from qbraid import GateModelResultData
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from metriq_gym.benchmarks import quops
from metriq_gym.benchmarks.quops import QUOPS, QUOPSData, QUOPSResult
from metriq_gym.local.device import LocalAerDevice
from metriq_gym.local.job import LocalAerJob
from metriq_gym.qplatform.job import QPUTime, qpu_time
from metriq_gym.schema_validator import validate_and_create_model


def _benchmark(**overrides):
    params = validate_and_create_model(
        {
            "benchmark_name": "QUOPS",
            "shapes": [{"width": 2, "size": 4}, {"width": 2, "size": 8}],
            "num_circuits": 3,
            "num_spam_circuits": 3,
            "shots": 128,
            "seed": 42,
            "bootstrap_samples": 100,
            "max_circuits_per_job": 4,
            **overrides,
        }
    )
    return QUOPS(Namespace(), params)


@pytest.fixture
def local_device(monkeypatch, tmp_path):
    monkeypatch.setenv("MGYM_LOCAL_SIMULATOR_CACHE_DIR", str(tmp_path))
    return LocalAerDevice(
        provider=Mock(), backend=AerSimulator(seed_simulator=61, max_parallel_threads=1)
    )


def _poll_local(benchmark, data):
    jobs = [LocalAerJob(job_id) for job_id in data.provider_job_ids]
    return benchmark.poll_handler(data, [job.result().data for job in jobs], jobs)


def _recorded_data(shapes=None):
    """Small complete acquisition with known counts, without compiler or provider mocks."""
    shapes = shapes or [{"width": 2, "size": 4}]
    records = []
    batches = []
    for index, shape in enumerate(shapes):
        for _ in range(2):
            for ensemble in ("mixed", "reference"):
                records.append(
                    {
                        "shape_index": index,
                        "width": shape["width"],
                        "ensemble": ensemble,
                        "ideal_bitstring": "1" + "0" * (shape["width"] - 1),
                    }
                )
        batches.append(
            {
                "shape_index": index,
                "num_circuits": 4,
                "provider_job_ids": [f"shape-{index}"],
                "execution_seconds": None,
                "timing_source": None,
            }
        )
    for width in sorted({shape["width"] for shape in shapes}):
        records.extend(
            {
                "shape_index": None,
                "width": width,
                "ensemble": "spam",
                "ideal_bitstring": "0" * width,
            }
            for _ in range(2)
        )
        batches.append(
            {
                "shape_index": None,
                "num_circuits": 2,
                "provider_job_ids": [f"spam-{width}"],
                "execution_seconds": None,
                "timing_source": None,
            }
        )
    data = QUOPSData(
        provider_job_ids=[job_id for batch in batches for job_id in batch["provider_job_ids"]],
        shapes=shapes,
        circuit_records=records,
        batches=batches,
        compilation={},
        seed=42,
        shots=100,
        bootstrap_samples=100,
    )
    counts = [{record["ideal_bitstring"]: 100} for record in records]
    return data, counts


def _results(counts):
    return [GateModelResultData(measurement_counts=counts)]


def test_local_dispatch_poll_and_serialized_replay(local_device):
    benchmark = _benchmark()
    data = benchmark.dispatch_handler(local_device)
    result = _poll_local(benchmark, data)

    assert result.quops == 8
    assert result.width == 2
    assert result.scan_status == "limit_reached"
    assert result.quops_per_second is None
    assert result.score.value == 8
    assert data.polarization_estimator == result.polarization_estimator == "mcfe"
    assert data.gate_set == result.gate_set == "pauli_rotations_cnot"
    assert all(row.polarization.value == pytest.approx(1) for row in result.shapes)
    assert all(row.certified and row.timing_source is None for row in result.shapes)
    assert len(data.circuit_records) == 15
    assert [batch["num_circuits"] for batch in data.batches] == [4, 2, 4, 2, 3]
    assert len(data.provider_job_ids) == 5

    restored_data = QUOPSData(**json.loads(json.dumps(asdict(data))))
    jobs = [LocalAerJob(job_id) for job_id in data.provider_job_ids]
    replay = benchmark.poll_handler(restored_data, [job.result().data for job in jobs], [])
    assert replay == result
    assert QUOPSResult.model_validate_json(result.model_dump_json()) == result


def test_local_noise_reduces_polarization(local_device):
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.08, 2), ["cx"])
    local_device._backend.set_options(noise_model=noise)
    benchmark = _benchmark(
        shapes=[{"width": 2, "size": 8}], num_circuits=12, num_spam_circuits=12, shots=2048
    )

    result = _poll_local(benchmark, benchmark.dispatch_handler(local_device))

    polarization = result.shapes[0].polarization
    assert 0.1 < polarization.value < 0.99
    assert polarization.uncertainty > 0
    assert result.quops_per_second is None


def test_resource_estimate_matches_dispatch_and_reproduces_hashes(local_device, monkeypatch):
    benchmark = _benchmark()
    batches, first = benchmark._build(local_device)
    _, second = _benchmark(polarization_estimator="mcfe", gate_set="pauli_rotations_cnot")._build(
        local_device
    )
    _, changed_seed = _benchmark(seed=43)._build(local_device)

    def hashes(data):
        return [record["circuit_sha256"] for record in data.circuit_records]

    assert hashes(first) == hashes(second)
    assert hashes(first) != hashes(changed_seed)
    assert first.compilation["layouts"]["2"]["physical_qubits"] == [0, 1]

    submit = Mock(side_effect=AssertionError("Estimation must not submit"))
    monkeypatch.setattr(quops, "submit_quops_circuits", submit)
    estimate = benchmark.estimate_resources_handler(local_device)
    assert [batch.circuits for batch in estimate] == [batch.circuits for batch in batches]
    assert sum(len(batch.circuits) * batch.shots for batch in estimate) == 15 * 128
    assert all(not batch.transpile for batch in estimate)
    assert all(not batch.sampled for batch in estimate)
    submit.assert_not_called()


def test_protocol_choices_default_to_published_protocol():
    default = _benchmark().params.model_dump()
    explicit = _benchmark(
        polarization_estimator="mcfe", gate_set="pauli_rotations_cnot"
    ).params.model_dump()

    assert default == explicit
    assert default["polarization_estimator"] == "mcfe"
    assert default["gate_set"] == "pauli_rotations_cnot"


def test_legacy_data_without_protocol_choices_replays_with_original_defaults():
    data, counts = _recorded_data()
    saved = asdict(data)
    saved.pop("polarization_estimator")
    saved.pop("gate_set")
    restored = QUOPSData(**json.loads(json.dumps(saved)))
    benchmark = _benchmark()
    # Polling uses the acquired protocol choices, independent of current parameters.
    benchmark.params = benchmark.params.model_copy(
        update={"polarization_estimator": "other", "gate_set": "other"}
    )

    result = benchmark.poll_handler(restored, _results(counts), [])

    assert result.quops == 4
    assert result.polarization_estimator == "mcfe"
    assert result.gate_set == "pauli_rotations_cnot"


@pytest.mark.parametrize("option", ["polarization_estimator", "gate_set"])
@pytest.mark.parametrize("handler", ["dispatch_handler", "estimate_resources_handler"])
def test_unsupported_protocol_choices_fail_before_compilation(monkeypatch, option, handler):
    benchmark = _benchmark()
    benchmark.params = benchmark.params.model_copy(update={option: "unsupported"})
    compile_circuit = Mock()
    submit = Mock()
    monkeypatch.setattr(quops, "compile_quops_circuit", compile_circuit)
    monkeypatch.setattr(quops, "submit_quops_circuits", submit)

    with pytest.raises(ValueError, match=f"Unsupported QUOPS {option}"):
        getattr(benchmark, handler)(Mock())

    compile_circuit.assert_not_called()
    submit.assert_not_called()


@pytest.mark.parametrize("option", ["polarization_estimator", "gate_set"])
def test_unsupported_saved_protocol_choices_cannot_be_reinterpreted(option):
    data, counts = _recorded_data()
    setattr(data, option, "unsupported")

    with pytest.raises(ValueError, match=f"Unsupported QUOPS {option}"):
        _benchmark().poll_handler(data, _results(counts), [])


@pytest.mark.parametrize("batch_limit, expected_jobs", [(1, 75), (7, 12), (100, 3)])
def test_fast_estimate_preserves_full_counts_and_batch_overheads(
    local_device, batch_limit, expected_jobs
):
    benchmark = _benchmark(num_circuits=13, num_spam_circuits=23, max_circuits_per_job=batch_limit)
    original_params = benchmark.params.model_dump()

    batches = benchmark.estimate_resources_handler(local_device)
    circuits = [circuit for batch in batches for circuit in batch.circuits]

    assert len(circuits) == 75  # Two shapes * 13 pairs, plus 23 SPAM circuits.
    assert sum(len(batch.circuits) * batch.shots for batch in batches) == 75 * 128
    assert len(batches) == expected_jobs
    assert all(0 < len(batch.circuits) <= batch_limit for batch in batches)
    assert len({id(circuit) for circuit in circuits}) == 30  # 2 * 5 pairs + 10 SPAM.
    assert all(batch.sampled and not batch.transpile for batch in batches)
    assert benchmark.params.model_dump() == original_params


def test_fast_estimate_samples_spam_separately_for_each_width(local_device):
    benchmark = _benchmark(
        shapes=[{"width": 2, "size": 4}, {"width": 3, "size": 10}],
        num_circuits=13,
        num_spam_circuits=23,
        max_circuits_per_job=7,
    )

    batches = benchmark.estimate_resources_handler(local_device)
    circuits = [circuit for batch in batches for circuit in batch.circuits]

    assert len(circuits) == 98  # Two shapes * 13 pairs, plus 23 SPAM per width.
    assert len(batches) == 16
    assert len({id(circuit) for circuit in circuits}) == 40
    assert sum(circuit.num_clbits == 2 for circuit in circuits) == 49
    assert sum(circuit.num_clbits == 3 for circuit in circuits) == 49


def test_fast_estimate_is_deterministic_without_changing_dispatch(local_device):
    benchmark = _benchmark(num_circuits=13, num_spam_circuits=23, max_circuits_per_job=7)
    original_params = benchmark.params.model_dump()
    first = benchmark.estimate_resources_handler(local_device)
    second = benchmark.estimate_resources_handler(local_device)
    assert [batch.circuits for batch in first] == [batch.circuits for batch in second]
    assert benchmark.params.model_dump() == original_params

    benchmark.args.full = True
    full = benchmark.estimate_resources_handler(local_device)
    dispatch_batches, dispatch_data = benchmark._build(local_device)
    assert [batch.circuits for batch in full] == [batch.circuits for batch in dispatch_batches]
    full_circuits = [circuit for batch in full for circuit in batch.circuits]
    assert len({id(circuit) for circuit in full_circuits}) == 75
    assert len(dispatch_data.circuit_records) == 75
    assert all(not batch.sampled for batch in full)
    assert benchmark.params.model_dump() == original_params


def test_boston_shape_usage_estimate_on_offline_cz_backend(monkeypatch):
    from math import isfinite

    from qbraid.programs import ExperimentType, ProgramSpec
    from qbraid.runtime import QiskitBackend, TargetProfile
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    from metriq_gym.qplatform.ibm_estimation import estimate_ibm_usage

    backend = FakeMarrakesh()
    profile = TargetProfile(
        device_id=backend.name,
        simulator=True,
        num_qubits=backend.num_qubits,
        experiment_type=ExperimentType.GATE_MODEL,
        program_spec=ProgramSpec(QuantumCircuit),
    )
    device = QiskitBackend(
        profile, service=SimpleNamespace(backend=lambda *args, **kwargs: backend)
    )
    sampler = Mock(side_effect=AssertionError("An offline estimate must not create a Sampler"))
    monkeypatch.setattr("qiskit_ibm_runtime.SamplerV2", sampler)
    benchmark = _benchmark(
        shapes=[{"width": 6, "size": 204}],
        num_circuits=2,
        num_spam_circuits=2,
        shots=1000,
        dynamical_decoupling=True,
    )

    batches = benchmark.estimate_resources_handler(device)
    seconds = estimate_ibm_usage(batches, device)

    assert isfinite(seconds) and seconds > 0
    assert sum(len(batch.circuits) * batch.shots for batch in batches) == 6000
    assert all(not batch.transpile for batch in batches)
    sampler.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"shapes": []},
        {"num_circuits": 1},
        {"num_spam_circuits": 1},
        {"shots": 0},
        {"physical_qubits": [0, 0]},
        {"polarization_estimator": "unsupported"},
        {"gate_set": "unsupported"},
        {"unknown_option": True},
    ],
)
def test_schema_rejects_invalid_parameters(overrides):
    with pytest.raises(ValidationError):
        _benchmark(**overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"shapes": [{"width": 2, "size": 16}]}, "width\\^2"),
        ({"shapes": [{"width": 2, "size": 6}]}, "representable"),
        (
            {"shapes": [{"width": 2, "size": 8}, {"width": 2, "size": 4}]},
            "increasing",
        ),
        ({"physical_qubits": [0, 1, 2]}, "match the width"),
    ],
)
def test_configuration_errors_precede_compilation_and_submission(monkeypatch, overrides, message):
    compile_circuit = Mock()
    submit = Mock()
    monkeypatch.setattr(quops, "compile_quops_circuit", compile_circuit)
    monkeypatch.setattr(quops, "submit_quops_circuits", submit)

    with pytest.raises(ValueError, match=message):
        _benchmark(**overrides).dispatch_handler(Mock())

    compile_circuit.assert_not_called()
    submit.assert_not_called()


def test_later_passing_shapes_do_not_reopen_failed_gate():
    data, counts = _recorded_data(
        [{"width": 2, "size": 4}, {"width": 2, "size": 8}, {"width": 3, "size": 10}]
    )
    for index, record in enumerate(data.circuit_records):
        if record["shape_index"] == 1 and record["ensemble"] == "mixed":
            counts[index] = {"00": 25, "01": 25, "10": 25, "11": 25}

    result = _benchmark().poll_handler(data, _results(counts), [])

    assert result.quops == 4
    assert result.scan_status == "threshold_reached"
    assert [row.passed for row in result.shapes] == [True, False, True]
    assert [row.certified for row in result.shapes] == [True, False, False]


def test_unresolved_reference_cannot_be_certified():
    data, counts = _recorded_data()
    for index, record in enumerate(data.circuit_records):
        if record["ensemble"] == "reference":
            counts[index] = {"00": 25, "01": 25, "10": 25, "11": 25}

    result = _benchmark().poll_handler(data, _results(counts), [])

    assert result.quops == 0
    assert result.width is None
    assert result.scan_status == "unresolved"
    assert result.shapes[0].polarization is None
    assert result.shapes[0].lower_bound is None
    assert not result.shapes[0].certified


@pytest.mark.parametrize(
    "replacement, message",
    [
        ({"10": 99}, "shot count"),
        ({"1": 100}, "target width"),
        ({"10": 101, "01": -1}, "nonnegative integers"),
        ({"10": 100.0}, "nonnegative integers"),
        ({}, "measurement counts"),
    ],
)
def test_malformed_counts_cannot_produce_a_score(replacement, message):
    data, counts = _recorded_data()
    counts[0] = replacement
    with pytest.raises(ValueError, match=message):
        _benchmark().poll_handler(data, _results(counts), [])


def test_missing_or_truncated_results_cannot_produce_a_score():
    data, counts = _recorded_data()
    for result_data in ([], _results(counts[:-1]), [GateModelResultData()]):
        with pytest.raises(ValueError, match="counts|count"):
            _benchmark().poll_handler(data, result_data, [])


@pytest.fixture
def timed_job():
    """A provider with its own measured-QPU timer, independent of IBM."""

    @dataclass
    class TimedJob:
        id: str
        seconds: float | None
        source: str = "test_qpu_usage"

    @qpu_time.register(TimedJob)
    def _(job):
        return QPUTime(job.seconds, job.source) if job.seconds is not None else None

    return TimedJob


def test_measured_timing_excludes_spam_and_survives_replay(timed_job):
    data, counts = _recorded_data()
    jobs = [timed_job("shape-0", 2), timed_job("spam-2", 100)]

    result = _benchmark().poll_handler(data, _results(counts), jobs)

    assert result.shapes[0].execution_seconds == 2
    assert result.shapes[0].timing_source == "test_qpu_usage"
    assert result.quops_per_second == pytest.approx(2 * 4 * 400 / 2)
    assert data.batches[0]["timing_source"] == "test_qpu_usage"
    restored = QUOPSData(**json.loads(json.dumps(asdict(data))))
    assert _benchmark().poll_handler(restored, _results(counts), []) == result


@pytest.mark.parametrize("missing_job", [True, False])
def test_incomplete_timing_stays_null(timed_job, missing_job):
    data, counts = _recorded_data()
    data.batches[0]["provider_job_ids"] = ["mirror-a", "mirror-b"]
    jobs = [timed_job("mirror-a", 2)]
    if not missing_job:
        jobs.append(timed_job("mirror-b", None))

    result = _benchmark().poll_handler(data, _results(counts), jobs)

    assert result.quops_per_second is None
    assert result.shapes[0].execution_seconds is None
    assert result.shapes[0].timing_source is None


def test_ibm_usage_produces_rate_without_provider_logic_in_benchmark():
    from qbraid.runtime import QiskitJob

    data, counts = _recorded_data()
    job = object.__new__(QiskitJob)
    job._job_id = "shape-0"
    job._job = SimpleNamespace(
        metrics=lambda: {"usage": {"qpu_charge_time_seconds": 2, "status": "complete"}}
    )

    result = _benchmark().poll_handler(data, _results(counts), [job])

    assert result.quops_per_second == pytest.approx(1600)
    assert result.shapes[0].timing_source == "ibm_qpu_usage"


def test_rate_sums_every_mirror_job_across_batches(timed_job):
    data, counts = _recorded_data()
    data.batches[0].update(num_circuits=2, provider_job_ids=["mirror-a", "mirror-b"])
    data.batches.insert(1, {**data.batches[0], "provider_job_ids": ["mirror-c"]})
    jobs = [timed_job("mirror-a", 1.25), timed_job("mirror-b", 0.75), timed_job("mirror-c", 2)]

    result = _benchmark().poll_handler(data, _results(counts), jobs)

    assert result.shapes[0].execution_seconds == 4
    assert result.quops_per_second == pytest.approx(800)


def test_winning_rate_uses_only_that_shapes_mirrors(timed_job):
    data, counts = _recorded_data([{"width": 2, "size": 4}, {"width": 2, "size": 8}])
    jobs = [timed_job("shape-0", 100), timed_job("shape-1", 2)]

    result = _benchmark().poll_handler(data, _results(counts), jobs)

    assert result.quops == 8
    assert result.shapes[0].quops_per_second == pytest.approx(32)
    assert result.quops_per_second == pytest.approx(3200)


def test_legacy_saved_ibm_timing_remains_replayable():
    data, counts = _recorded_data()
    data.batches[0].pop("timing_source")
    data.batches[0]["execution_seconds"] = 2

    result = _benchmark().poll_handler(data, _results(counts), [])

    assert result.quops_per_second == pytest.approx(1600)
    assert result.shapes[0].timing_source == "ibm_qpu_usage"


@pytest.mark.parametrize("seconds", [None, 0, -1, float("nan"), float("inf"), True])
def test_invalid_saved_timing_does_not_produce_rate(seconds):
    data, counts = _recorded_data()
    data.batches[0].update(execution_seconds=seconds, timing_source="test_qpu_usage")

    result = _benchmark().poll_handler(data, _results(counts), [])

    assert result.quops == 4
    assert result.quops_per_second is None
    assert result.shapes[0].execution_seconds is None


def test_explicitly_missing_timing_source_is_not_assumed_ibm():
    data, counts = _recorded_data()
    data.batches[0]["execution_seconds"] = 2

    result = _benchmark().poll_handler(data, _results(counts), [])

    assert result.quops_per_second is None
    assert result.shapes[0].timing_source is None


def test_incompatible_timing_sources_across_batches_do_not_produce_rate(timed_job):
    data, counts = _recorded_data()
    data.batches[0]["num_circuits"] = 2
    data.batches.insert(1, {**data.batches[0], "provider_job_ids": ["other"]})
    jobs = [timed_job("shape-0", 2), timed_job("other", 3, "different_timer")]

    result = _benchmark().poll_handler(data, _results(counts), jobs)

    assert result.quops_per_second is None
    assert result.shapes[0].timing_source is None


def test_unknown_protocol_cannot_be_replayed():
    data, counts = _recorded_data()
    data.protocol = "future-mcfe-v2"
    with pytest.raises(ValueError, match="Unsupported QUOPS protocol"):
        _benchmark().poll_handler(data, _results(counts), [])
