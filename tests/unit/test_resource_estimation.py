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
    [
        ("quantinuum", True, False),
        ("ibm", False, True),
        ("local", False, False),
        ("aws", False, False),
        ("braket", False, False),
    ],
)
def test_print_estimate_only_shows_selected_provider_costs(provider, show_hqc, show_ibm, capsys):
    # Even an estimate carrying both fields must only display the chosen provider's units.
    print_resource_estimate("test", provider, "device", _estimate(seconds=90), verbose=True)

    output = capsys.readouterr().out
    assert ("Total HQCs" in output) is show_hqc
    assert ("HQC per circuit" in output) is show_hqc
    assert ("Estimated IBM QPU time (s)" in output) is show_ibm
    assert "QPU time (min)" not in output
    assert "Total shots" in output
    assert ("Total tasks" in output) is (provider in {"aws", "braket"})


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

    print_resource_estimate("Suite total", "ibm", "device", total, verbose=True)

    output = capsys.readouterr().out
    assert "Partial IBM QPU time (s)" in output
    assert "Estimated IBM QPU time (s)" not in output
    assert "excludes entries without timing" in output
    assert "Missing timing data" in output
    assert "HQC" not in output


def test_print_unavailable_ibm_time(capsys):
    print_resource_estimate("test", "ibm", None, _estimate(complete=False), verbose=True)

    output = capsys.readouterr().out
    assert "Estimated IBM QPU time (s)" in output
    assert "n/a" in output
    assert "incomplete" in output


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("ibm", "Estimated IBM QPU time (s): 90.000\n"),
        ("quantinuum", "Total HQCs: 5.22\n"),
        ("aws", "Total shots: 100\nTotal tasks: 1\n"),
        ("braket", "Total shots: 100\nTotal tasks: 1\n"),
        ("local", "No cost estimate for local. Use --verbose for circuit resources.\n"),
    ],
)
def test_default_output_has_header_and_provider_summary(provider, expected, capsys):
    print_resource_estimate(
        "test", provider, "device", _estimate(seconds=90, notes=["Verbose assumption"])
    )

    assert capsys.readouterr().out == f"Resource estimate for test on {provider}:device\n{expected}"


@pytest.mark.parametrize("provider", ["aws", "braket", " AWS ", " Braket "])
@pytest.mark.parametrize("verbose", [False, True])
def test_aws_counts_each_circuit_as_a_task(provider, verbose, capsys):
    circuit = QuantumCircuit(1)
    estimate = aggregate_resource_estimates(
        [CircuitBatch([circuit, circuit], shots=1_000), CircuitBatch([circuit], shots=2_500)]
    )
    assert estimate.job_count == 2

    print_resource_estimate("test", provider, "device", estimate, verbose=verbose)

    output = capsys.readouterr().out
    if verbose:
        rows = [
            tuple(cell.strip() for cell in line.strip("|").split("|"))
            for line in output.splitlines()
        ]
        assert ("Total shots", "4_500") in rows
        assert ("Total tasks", "3") in rows
        assert ("Jobs", "2") in rows
        assert ("Circuits", "3") in rows
    else:
        assert output == (
            f"Resource estimate for test on {provider}:device\nTotal shots: 4,500\nTotal tasks: 3\n"
        )


@pytest.mark.parametrize("seconds", [None, 12.5])
def test_compact_output_marks_missing_or_partial_timing(seconds, capsys):
    print_resource_estimate("test", "ibm", "device", _estimate(seconds=seconds, complete=False))

    output = capsys.readouterr().out
    assert len(output.splitlines()) == 2
    assert output.splitlines()[0] == "Resource estimate for test on ibm:device"
    assert "--verbose" in output
    if seconds is None:
        assert "n/a" in output
    else:
        assert "Partial IBM QPU time (s): 12.500" in output
