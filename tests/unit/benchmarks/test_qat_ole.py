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
    result = QATOLEResult(expectation_value=BenchmarkScore(value=0.85, uncertainty=0.02))
    assert result.score.value == pytest.approx(0.85)
    assert result.score.uncertainty == pytest.approx(0.02)


def test_qat_ole_result_values_and_uncertainties():
    result = QATOLEResult(expectation_value=BenchmarkScore(value=0.72, uncertainty=0.03))
    assert result.values == pytest.approx({"expectation_value": 0.72})
    assert result.uncertainties == pytest.approx({"expectation_value": 0.03})


def test_qat_ole_result_exporter_payload():
    job = _build_metriq_job()
    result = QATOLEResult(expectation_value=BenchmarkScore(value=0.91, uncertainty=0.01))
    exporter = _DummyExporter(job, result)

    payload = exporter.as_dict()
    assert payload["results"]["expectation_value"]["value"] == pytest.approx(0.91)
    assert payload["results"]["expectation_value"]["uncertainty"] == pytest.approx(0.01)
    assert payload["results"]["score"]["value"] == pytest.approx(0.91)
    assert payload["platform"] == {"provider": "provider", "device": "device"}


def test_qat_ole_result_score_keys_match():
    result = QATOLEResult(expectation_value=BenchmarkScore(value=0.5, uncertainty=0.05))
    assert set(result.values.keys()) == {"expectation_value"}
    assert set(result.uncertainties.keys()) == {"expectation_value"}
