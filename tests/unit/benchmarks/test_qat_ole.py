from datetime import datetime

import pytest

from metriq_gym.benchmarks.benchmark import BenchmarkScore
from metriq_gym.benchmarks.qat_ole import (
    QATOLEResult,
    _pauli_z_product_expectation,
)
from metriq_gym.constants import JobType
from metriq_gym.exporters.base_exporter import BaseExporter
from metriq_gym.job_manager import MetriqGymJob


class _DummyExporter(BaseExporter):
    def export(self) -> None:  # pragma: no cover
        raise NotImplementedError


def _build_metriq_job() -> MetriqGymJob:
    return MetriqGymJob(
        id="test-job",
        job_type=JobType.QAT_OLE,
        params={"benchmark_name": "QAT OLE", "circuit": "49Q_L3", "shots": 1000},
        data={"provider_job_ids": ["qid"], "observable_qubits": [52, 59, 72], "shots": 1000},
        provider_name="provider",
        device_name="device",
        dispatch_time=datetime.now(),
    )


def _make_result(value: float = 0.85, uncertainty: float = 0.02) -> QATOLEResult:
    return QATOLEResult(
        observable_value=BenchmarkScore(value=value, uncertainty=uncertainty),
        shots=1000,
        circuit_id="49Q_L3",
        num_qubits=156,
        num_gates=2048,
    )


# --- Pauli-Z product expectation ---


def test_pauli_z_product_expectation_all_even():
    # All outcomes have even parity: expectation = 1.0
    counts = {"000": 500, "011": 300, "101": 150, "110": 50}
    value, uncertainty = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(1.0)
    assert uncertainty == pytest.approx(0.0)


def test_pauli_z_product_expectation_all_odd():
    # All outcomes have odd parity: expectation = -1.0
    counts = {"001": 400, "010": 300, "100": 200, "111": 100}
    value, uncertainty = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(-1.0)
    assert uncertainty == pytest.approx(0.0)


def test_pauli_z_product_expectation_equal_split():
    # Equal even/odd split: expectation = 0.0
    counts = {"000": 500, "001": 500}
    value, uncertainty = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(0.0)
    assert uncertainty == pytest.approx(2.0 * (0.5 * 0.5 / 1000) ** 0.5)


def test_pauli_z_product_expectation_empty_counts():
    value, uncertainty = _pauli_z_product_expectation({})
    assert value == pytest.approx(0.0)
    assert uncertainty == pytest.approx(0.0)


def test_pauli_z_product_expectation_single_qubit_one():
    # Single-qubit case: outcome "1" is odd, "0" is even
    counts = {"0": 700, "1": 300}
    value, uncertainty = _pauli_z_product_expectation(counts)
    # P(even) = 0.7, expectation = 2*0.7 - 1 = 0.4
    assert value == pytest.approx(0.4)
    assert uncertainty == pytest.approx(2.0 * (0.7 * 0.3 / 1000) ** 0.5)


def test_pauli_z_product_expectation_ghz_two_qubit():
    # GHZ state on 2 qubits: |00> + |11>; both outcomes have even parity
    counts = {"00": 512, "11": 488}
    value, uncertainty = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(1.0)
    assert uncertainty == pytest.approx(0.0)


# --- QATOLEResult model ---


def test_qat_ole_result_score_properties():
    result = _make_result(0.85, 0.02)
    assert result.score.value == pytest.approx(0.85)
    assert result.score.uncertainty == pytest.approx(0.02)


def test_qat_ole_result_extra_fields():
    result = _make_result(0.72, 0.03)
    assert result.shots == 1000
    assert result.circuit_id == "49Q_L3"
    assert result.num_qubits == 156
    assert result.num_gates == 2048


def test_qat_ole_result_values_and_uncertainties():
    result = _make_result(0.72, 0.03)
    # observable_value is a BenchmarkScore; shots/num_qubits/num_gates are bare ints
    assert result.values["observable_value"] == pytest.approx(0.72)
    assert result.uncertainties["observable_value"] == pytest.approx(0.03)
    assert result.values["shots"] == pytest.approx(1000)
    assert result.values["num_qubits"] == pytest.approx(156)
    assert result.values["num_gates"] == pytest.approx(2048)


def test_qat_ole_result_exporter_payload():
    job = _build_metriq_job()
    result = _make_result(0.91, 0.01)
    exporter = _DummyExporter(job, result)

    payload = exporter.as_dict()
    assert payload["results"]["observable_value"]["value"] == pytest.approx(0.91)
    assert payload["results"]["observable_value"]["uncertainty"] == pytest.approx(0.01)
    assert payload["results"]["score"]["value"] == pytest.approx(0.91)
    assert payload["platform"] == {"provider": "provider", "device": "device"}


def test_qat_ole_result_score_keys_match():
    result = _make_result(0.5, 0.05)
    assert "observable_value" in result.values
    assert "circuit_id" not in result.values  # string field, not a metric
    assert result.uncertainties.get("observable_value") is not None
