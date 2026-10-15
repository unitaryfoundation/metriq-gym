from dataclasses import dataclass, field

import pytest
from qiskit import QuantumCircuit

from metriq_gym.resource_estimation import (
    CircuitBatch,
    aggregate_resource_estimates,
    merge_resource_estimates,
    print_resource_estimate,
    quantinuum_hqc_formula,
)


@dataclass
class _FakeProfile:
    basis_gates: list[str] | None = None


@dataclass
class _FakeDevice:
    id: str = "test-simulator"
    profile: _FakeProfile = field(default_factory=_FakeProfile)
    num_qubits: int | None = None


def _estimate(*, seconds=None, complete=True, notes=()):
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    estimate = aggregate_resource_estimates(
        [CircuitBatch([circuit], shots=100, notes=list(notes))],
        hqc_fn=quantinuum_hqc_formula,
    )
    estimate.ibm_quantum_seconds = seconds
    estimate.ibm_timing_complete = complete
    return estimate


@pytest.mark.parametrize(
    ("provider", "show_hqc", "show_ibm"),
    [("quantinuum", True, False), ("ibm", False, True), ("local", False, False)],
)
def test_print_estimate_only_shows_selected_provider_costs(provider, show_hqc, show_ibm, capsys):
    # Even an estimate carrying both fields must only display the chosen provider's units.
    print_resource_estimate("test", provider, "device", _estimate(seconds=90))

    output = capsys.readouterr().out
    assert ("Total HQCs" in output) is show_hqc
    assert ("HQC per circuit" in output) is show_hqc
    assert ("Estimated IBM QPU time (s)" in output) is show_ibm
    assert ("Estimated IBM QPU time (min)" in output) is show_ibm
    assert "Total shots" in output


@pytest.mark.parametrize(
    ("seconds", "complete", "expected", "expected_complete"),
    [
        ([20.0, 40.0], [True, True], 60.0, True),
        ([20.0, None], [True, False], 20.0, False),
        ([None, 0.0], [False, True], 0.0, False),
        ([None, None], [False, False], None, False),
    ],
)
def test_merge_preserves_ibm_timing_coverage(seconds, complete, expected, expected_complete):
    entries = [
        _estimate(seconds=value, complete=ok, notes=["Shared assumption"])
        for value, ok in zip(seconds, complete)
    ]

    total = merge_resource_estimates(entries)

    assert total.ibm_quantum_seconds == expected
    assert total.ibm_timing_complete is expected_complete
    assert total.total_shots == 200
    assert total.notes == ["Shared assumption"]
    assert [entry.ibm_quantum_seconds for entry in entries] == seconds


def test_print_partial_ibm_time_and_exclusions(capsys):
    total = merge_resource_estimates(
        [_estimate(seconds=20), _estimate(complete=False, notes=["Missing timing data"])]
    )

    print_resource_estimate("Suite total", "ibm", "device", total)

    output = capsys.readouterr().out
    assert "Partial IBM QPU time (s)" in output
    assert "Estimated IBM QPU time (s)" not in output
    assert "excludes entries without timing" in output
    assert "Missing timing data" in output
    assert "HQC" not in output


def test_print_unavailable_ibm_time(capsys):
    print_resource_estimate("test", "ibm", None, _estimate(complete=False))

    output = capsys.readouterr().out
    assert "Estimated IBM QPU time (s)" in output
    assert "n/a" in output
    assert "incomplete" in output
