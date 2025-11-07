import math
from datetime import datetime
from types import SimpleNamespace

import pytest
from qiskit import qasm3

from metriq_gym.benchmarks.wit import (
    WITResult,
    WormholeTeleportationConfig,
    build_wit_config_from_params,
    legacy_wit_circuit,
    wit_circuit,
)
from pydantic import Field
from metriq_gym.benchmarks.benchmark import BenchmarkScore, BenchmarkResult
from metriq_gym.constants import JobType
from metriq_gym.helpers.statistics import (
    binary_expectation_stddev,
    binary_expectation_value,
)
from metriq_gym.exporters.base_exporter import BaseExporter
from metriq_gym.job_manager import MetriqGymJob

BASE_Z_ANGLES = (0.0283397, 0.00519953, 0.0316079)


class _DummyExporter(BaseExporter):
    def export(self) -> None:  # pragma: no cover - not used in tests
        raise NotImplementedError


def _build_metriq_job() -> MetriqGymJob:
    return MetriqGymJob(
        id="test-job",
        job_type=JobType.WIT,
        params={
            "benchmark_name": "WIT",
            "shots": 10,
            "n_qubits_per_side": 3,
            "message_size": 1,
            "insert_message_method": "reset",
            "interaction_coupling_strength": 1.5707963267948966,
            "x_rotation_transverse_angle": 0.7853981633974483,
            "zz_rotation_angle": 0.7853981633974483,
            "z_rotation_angles": [0.0283397, 0.00519953, 0.0316079],
            "time_steps": 3,
            "num_qubits": 6,
        },
        data={"provider_job_ids": ["qid"]},
        provider_name="provider",
        device_name="device",
        dispatch_time=datetime.now(),
    )


def test_calculate_expectation_value_uses_effective_shots():
    counts = {"1": 6, "0": 4}
    assert binary_expectation_value(10, counts) == pytest.approx(0.6)

    counts_truncated = {"1": 3}
    # Only three total shots should be used to avoid inflating the expectation value.
    assert binary_expectation_value(10, counts_truncated) == pytest.approx(1.0)


def test_calculate_expectation_value_handles_zero_counts():
    assert binary_expectation_value(100, {}) == 0.0


def test_calculate_expectation_value_error_binomial_uncertainty():
    counts = {"1": 60, "0": 40}
    err = binary_expectation_stddev(100, counts)
    assert err == pytest.approx(0.049, abs=1e-3)


def test_calculate_expectation_value_error_handles_zero_counts():
    assert binary_expectation_stddev(100, {}) == 0.0


def test_wit_result_exports_symmetric_results_and_uncertainties():
    job = _build_metriq_job()
    result = WITResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))
    exporter = _DummyExporter(job, result)

    payload = exporter.as_dict()
    assert payload["results"]["values"] == {"expectation_value": pytest.approx(0.5)}
    assert payload["results"]["uncertainties"] == {"expectation_value": pytest.approx(0.05)}
    assert payload["platform"] == {"provider": "provider", "device": "device"}
    assert result.values == pytest.approx({"expectation_value": 0.5})
    assert result.uncertainties == pytest.approx({"expectation_value": 0.05})


def test_wit_result_includes_score_in_export():
    job = _build_metriq_job()
    result = WITResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))
    exporter = _DummyExporter(job, result)

    payload = exporter.as_dict()
    assert payload["results"]["score"] == pytest.approx(0.5)


def test_wit_result_score_properties():
    r = WITResult(expectation_value=BenchmarkScore(value=0.33, uncertainty=0.01))
    assert r.score == pytest.approx(0.33)


def test_missing_direction_no_longer_raises_validation_error():
    class DummyResult(BenchmarkResult):
        metric: BenchmarkScore

        def compute_score(self):
            return None

    r = DummyResult(metric=BenchmarkScore(value=1.0, uncertainty=0.0))
    assert r.values["metric"] == pytest.approx(1.0)


def test_metadata_is_optional_and_ignored_for_scoring():
    class DummyResult2(BenchmarkResult):
        latency: BenchmarkScore = Field(...)

        def compute_score(self):
            return None

    r = DummyResult2(latency=BenchmarkScore(value=12.3, uncertainty=0.5))
    assert r.values["latency"] == pytest.approx(12.3)


def test_wit_result_uncertainty_keys_match_values():
    r = WITResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))
    assert set(r.values.keys()) == {"expectation_value"}
    assert set(r.uncertainties.keys()) == {"expectation_value"}


@pytest.mark.parametrize(
    ("total_qubits", "method"),
    [(6, "reset"), (7, "swap")],
)
def test_generalized_circuit_matches_legacy(total_qubits: int, method: str):
    params = SimpleNamespace(
        benchmark_name="WIT",
        shots=1000,
        n_qubits_per_side=3,
        message_size=1,
        insert_message_method=method,
        interaction_coupling_strength=math.pi / 2,
        x_rotation_transverse_angle=math.pi / 4,
        zz_rotation_angle=math.pi / 4,
        z_rotation_angles=list(BASE_Z_ANGLES),
        time_steps=3,
        num_qubits=total_qubits,
    )
    config = build_wit_config_from_params(params)
    assert config.total_qubits == total_qubits
    new_qasm = qasm3.dumps(wit_circuit(config))
    legacy_qasm = qasm3.dumps(legacy_wit_circuit(total_qubits))
    assert new_qasm == legacy_qasm


def test_swap_insertion_adds_ancilla_and_measures_right_side():
    config = WormholeTeleportationConfig(
        n_qubits_per_side=3,
        message_size=1,
        x_rotation_transverse_angle=math.pi / 4,
        zz_rotation_angle=math.pi / 4,
        z_rotation_angles=BASE_Z_ANGLES,
        time_steps=3,
        insert_message_method="swap",
        interaction_coupling_strength=math.pi / 2,
    )
    circuit = wit_circuit(config)
    assert circuit.num_qubits == 7
    measure_instruction = next(inst for inst in circuit.data if inst.operation.name == "measure")
    measured_qubit = measure_instruction.qubits[0]
    assert circuit.find_bit(measured_qubit).index == 5


def test_reset_insertion_preserves_expected_measurement():
    config = WormholeTeleportationConfig(
        n_qubits_per_side=3,
        message_size=1,
        x_rotation_transverse_angle=math.pi / 4,
        zz_rotation_angle=math.pi / 4,
        z_rotation_angles=BASE_Z_ANGLES,
        time_steps=3,
        insert_message_method="reset",
        interaction_coupling_strength=math.pi / 2,
    )
    circuit = wit_circuit(config)
    assert circuit.num_qubits == 6
    measure_instruction = next(inst for inst in circuit.data if inst.operation.name == "measure")
    measured_qubit = measure_instruction.qubits[0]
    assert circuit.find_bit(measured_qubit).index == 5


def test_build_wit_config_from_params_detects_num_qubits_mismatch():
    params = SimpleNamespace(
        benchmark_name="WIT",
        shots=1000,
        n_qubits_per_side=3,
        message_size=1,
        insert_message_method="swap",
        interaction_coupling_strength=math.pi / 2,
        x_rotation_transverse_angle=math.pi / 4,
        zz_rotation_angle=math.pi / 4,
        z_rotation_angles=list(BASE_Z_ANGLES),
        time_steps=3,
        num_qubits=6,
    )
    with pytest.raises(ValueError):
        build_wit_config_from_params(params)


def test_wormhole_config_rejects_swap_with_large_message():
    with pytest.raises(ValueError):
        WormholeTeleportationConfig(
            n_qubits_per_side=3,
            message_size=2,
            x_rotation_transverse_angle=math.pi / 4,
            zz_rotation_angle=math.pi / 4,
            z_rotation_angles=BASE_Z_ANGLES,
            time_steps=3,
            insert_message_method="swap",
            interaction_coupling_strength=math.pi / 2,
        )
