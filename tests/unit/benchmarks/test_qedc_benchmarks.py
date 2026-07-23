"""Tests for the QED-C benchmark wrapper, focused on QFT's fixed qubit count."""

import pytest

from metriq_gym.benchmarks.qedc_benchmarks import (
    calculate_accuracy_score,
    get_circuits_and_metrics,
)


def test_qft_num_qubits_builds_single_qubit_group():
    """A QFT run with a fixed num_qubits produces one QED-C qubit group."""
    circuits, circuit_metrics, circuit_identifiers = get_circuits_and_metrics(
        benchmark_name="Quantum Fourier Transform",
        params={
            "num_qubits": 4,
            "shots": 100,
            "max_circuits": 2,
            "method": 1,
            "use_midcircuit_measurement": False,
        },
    )

    # Exactly one qubit group, and it is the requested size.
    assert list(circuit_metrics.keys()) == ["4"]
    # Every generated circuit belongs to that single group.
    assert {num_qubits for num_qubits, _ in circuit_identifiers} == {"4"}
    assert len(circuits) == len(circuit_identifiers)
    assert len(circuits) >= 1


def test_qft_num_qubits_respects_max_circuits():
    """num_qubits leaves the per-group circuit count under max_circuits control."""
    _, circuit_metrics, circuit_identifiers = get_circuits_and_metrics(
        benchmark_name="Quantum Fourier Transform",
        params={
            "num_qubits": 5,
            "shots": 100,
            "max_circuits": 1,
            "method": 1,
            "use_midcircuit_measurement": False,
        },
    )

    assert list(circuit_metrics.keys()) == ["5"]
    assert len(circuit_identifiers) == 1


def test_accuracy_score_does_not_accumulate_across_calls():
    """Scoring several QED-C jobs in one process must not aggregate across them.

    QED-C's metrics module keeps group statistics in module-global state that
    aggregate_metrics() appends to. A suite that sweeps QFT over qubit counts
    polls multiple jobs in a single process, so each score must depend only on
    its own circuit metrics. Without the reset, the second and third calls would
    return inflated scores (2.0, 3.0) instead of 1.0.
    """
    for num_qubits in ("3", "4", "5"):
        circuit_metrics = {num_qubits: {str(i): {"fidelity": 1.0} for i in range(3)}}
        score, uncertainty = calculate_accuracy_score(circuit_metrics)
        assert score == pytest.approx(1.0)
        assert uncertainty == pytest.approx(0.0)
