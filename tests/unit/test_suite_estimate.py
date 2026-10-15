import argparse
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qiskit import QuantumCircuit

from metriq_gym import run
from metriq_gym.constants import JobType
from metriq_gym.resource_estimation import CircuitBatch, GateCounts, print_resource_estimate
from metriq_gym.suite_parser import parse_suite_file


def _entry(name, *, component=None, shots=10, num_qubits=6):
    return {
        "name": name,
        "component": component,
        "config": {"benchmark_name": "WIT", "num_qubits": num_qubits, "shots": shots},
    }


def _write_suite(tmp_path, entries=None, *, guarded=False):
    suite = {
        "name": "test_suite",
        "benchmarks": entries if entries is not None else [_entry("first"), _entry("second")],
    }
    if guarded:
        suite["full_suite_warning"] = "This test suite is expensive."
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    return str(path)


def _args(suite_config, **kwargs):
    return argparse.Namespace(
        **{
            "suite_config": suite_config,
            "provider": "ibm",
            "device": "ibm_phoenix",
            "components": None,
            "all_components": False,
            **kwargs,
        }
    )


@pytest.fixture
def estimation(monkeypatch):
    device = SimpleNamespace(id="ibm_phoenix", num_qubits=200, run=MagicMock())
    setup_device = MagicMock(return_value=device)
    registry = SimpleNamespace(get_available_benchmarks=lambda: [job.value for job in JobType])
    handler = MagicMock()
    handler.estimate_resources_handler.return_value = [
        CircuitBatch(circuits=[QuantumCircuit(6)], shots=10)
    ]

    def estimate_resources(_device):
        print("benchmark chatter")
        return handler.estimate_resources_handler.return_value

    def setup_benchmark_handler(*_args, **_kwargs):
        print("benchmark setup chatter")
        return handler

    handler.estimate_resources_handler.side_effect = estimate_resources
    setup_benchmark = MagicMock(side_effect=setup_benchmark_handler)
    printer = MagicMock()
    job_manager = MagicMock()
    new_job = MagicMock()

    def estimate_ibm_usage(batches, device):
        if device is None:
            raise ValueError("IBM QPU time estimation requires --device")
        return 12.5

    ibm_usage = MagicMock(side_effect=estimate_ibm_usage)
    monkeypatch.setattr(run, "setup_device", setup_device)
    monkeypatch.setattr(run, "_lazy_registry", lambda: registry)
    monkeypatch.setattr(run, "setup_benchmark", setup_benchmark)
    monkeypatch.setattr(run, "print_resource_estimate", printer)
    monkeypatch.setattr(run, "JobManager", job_manager)
    monkeypatch.setattr(run, "_new_job", new_job)
    monkeypatch.setattr("metriq_gym.qplatform.ibm_estimation.estimate_ibm_usage", ibm_usage)
    yield SimpleNamespace(
        device=device,
        setup_device=setup_device,
        handler=handler,
        setup_benchmark=setup_benchmark,
        printer=printer,
        ibm_usage=ibm_usage,
    )
    handler.dispatch_handler.assert_not_called()
    job_manager.assert_not_called()
    new_job.assert_not_called()
    device.run.assert_not_called()


def test_suite_estimate_sums_real_circuits_and_preserves_entry_estimates(
    tmp_path, estimation, capsys
):
    first = QuantumCircuit(2, 2)
    first.h(0)
    first.cx(0, 1)
    first.reset(1)
    first.measure([0, 1], [0, 1])
    second = QuantumCircuit(3, 3)
    second.x(0)
    second.ccx(0, 1, 2)
    second.measure([0, 1, 2], [0, 1, 2])
    estimation.handler.estimate_resources_handler.side_effect = [
        [CircuitBatch(circuits=[first, second], shots=10)],
        [CircuitBatch(circuits=[first], shots=20), CircuitBatch(circuits=[second], shots=30)],
    ]
    args = _args(_write_suite(tmp_path), provider="quantinuum", device="H1-1", verbose=True)

    run.estimate_suite(args)

    estimation.setup_device.assert_called_once_with("quantinuum", "H1-1")
    assert estimation.handler.estimate_resources_handler.call_count == 2
    for call in estimation.handler.estimate_resources_handler.call_args_list:
        assert call.args == (estimation.device,)
    assert estimation.printer.call_count == 3
    assert all(call.kwargs == {"verbose": True} for call in estimation.printer.call_args_list)
    entry_estimates = [call.args[3] for call in estimation.printer.call_args_list[:-1]]
    total_label, provider, device_id, total = estimation.printer.call_args.args
    assert "Suite total" in total_label
    assert (provider, device_id) == ("quantinuum", "H1-1")
    assert total.job_count == 3
    assert total.circuit_count == 4
    assert total.total_shots == 70
    assert total.max_qubits == 3
    assert total.total_gate_counts == GateCounts(
        one_qubit=4, two_qubit=2, multi_qubit=2, measurements=10, resets=2
    )
    assert total.hqc_total == pytest.approx(20 + (10 * 36 + 10 * 31 + 20 * 36 + 30 * 31) / 5000)
    assert [(c.job_index, c.circuit_index) for c in total.per_circuit] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
    ]
    assert entry_estimates[0].total_gate_counts == GateCounts(
        one_qubit=2, two_qubit=1, multi_qubit=1, measurements=5, resets=1
    )
    assert [c.job_index for c in entry_estimates[1].per_circuit] == [0, 1]
    assert "2/2" in capsys.readouterr().out


def test_suite_estimate_without_device_prints_entry_and_total_tables(
    tmp_path, estimation, monkeypatch, capsys
):
    monkeypatch.setattr(run, "print_resource_estimate", print_resource_estimate)

    run.estimate_suite(_args(_write_suite(tmp_path), device=None, verbose=True))

    estimation.setup_device.assert_not_called()
    assert estimation.handler.estimate_resources_handler.call_count == 2
    estimation.handler.estimate_resources_handler.assert_called_with(None)
    output = capsys.readouterr().out
    assert "without device-specific topology" in output
    assert "(no device)" in output
    assert "Suite total" in output
    assert output.count("Total shots") == 3
    assert "first" in output
    assert "second" in output
    assert "HQC" not in output
    assert "IBM QPU time" in output
    assert "n/a" in output
    assert "--device" in output


@pytest.mark.parametrize("provider", ["ibm", "quantinuum", "local", "aws", "braket"])
def test_suite_estimate_uses_only_selected_provider_usage(
    tmp_path, estimation, monkeypatch, capsys, provider
):
    printer = MagicMock(wraps=print_resource_estimate)
    monkeypatch.setattr(run, "print_resource_estimate", printer)

    run.estimate_suite(_args(_write_suite(tmp_path), provider=provider))

    total = printer.call_args.args[3]
    printer.assert_called_once()
    assert printer.call_args.kwargs == {"verbose": False}
    assert total.circuit_count == 2
    assert total.total_shots == 20
    output = capsys.readouterr().out
    if provider == "ibm":
        assert estimation.ibm_usage.call_count == 2
        for call in estimation.ibm_usage.call_args_list:
            assert call.args[1] is estimation.device
        assert total.ibm_quantum_seconds == 25.0
        assert total.ibm_timing_complete
        assert output == (
            "Cost estimate for Suite total (test_suite) on ibm:ibm_phoenix\n"
            "Estimated IBM QPU time (s): 25.000\n"
        )
    else:
        estimation.ibm_usage.assert_not_called()
        assert total.ibm_quantum_seconds is None
        assert "IBM QPU time" not in output
    if provider == "quantinuum":
        assert total.hqc_total is not None
        assert total.hqc_total > 0
        assert output == (
            "Cost estimate for Suite total (test_suite) on quantinuum:ibm_phoenix\n"
            "Total HQCs: 10.12\n"
        )
    else:
        assert total.hqc_total is None
        assert all(circuit.hqc is None for circuit in total.per_circuit)
        assert "HQC" not in output
    if provider in {"aws", "braket"}:
        assert output == (
            f"Cost estimate for Suite total (test_suite) on {provider}:ibm_phoenix\n"
            "Total shots: 20\nTotal tasks: 2\n"
        )
    else:
        assert "Total shots" not in output
    assert "benchmark chatter" not in output
    assert "benchmark setup chatter" not in output
    assert "Estimating" not in output
    assert "(min)" not in output


@pytest.mark.parametrize("verbose", [False, True])
def test_suite_estimate_aws_aggregates_shots_and_circuit_tasks(
    tmp_path, estimation, monkeypatch, capsys, verbose
):
    printer = MagicMock(wraps=print_resource_estimate)
    monkeypatch.setattr(run, "print_resource_estimate", printer)
    circuit = QuantumCircuit(6)
    estimation.handler.estimate_resources_handler.side_effect = [
        [CircuitBatch(circuits=[circuit, circuit], shots=10)],
        [CircuitBatch(circuits=[circuit], shots=20), CircuitBatch(circuits=[circuit], shots=30)],
    ]

    run.estimate_suite(_args(_write_suite(tmp_path), provider="aws", verbose=verbose))

    total = printer.call_args.args[3]
    assert total.job_count == 3
    assert total.circuit_count == 4
    assert total.total_shots == 70
    estimation.ibm_usage.assert_not_called()
    output = capsys.readouterr().out
    if verbose:
        assert printer.call_count == 3
        assert output.count("Total tasks") == 3
        assert output.count("Total shots") == 3
        assert "Max qubits" in output
        assert "Suite total" in output
    else:
        printer.assert_called_once()
        assert output == (
            "Cost estimate for Suite total (test_suite) on aws:ibm_phoenix\n"
            "Total shots: 70\nTotal tasks: 4\n"
        )
    assert "HQC" not in output
    assert "IBM QPU time" not in output


def test_suite_estimate_verbose_prints_entries_and_total(tmp_path, estimation, monkeypatch, capsys):
    printer = MagicMock(wraps=print_resource_estimate)
    monkeypatch.setattr(run, "print_resource_estimate", printer)

    run.estimate_suite(_args(_write_suite(tmp_path), verbose=True))

    assert printer.call_count == 3
    assert all(call.kwargs == {"verbose": True} for call in printer.call_args_list)
    output = capsys.readouterr().out
    assert output.count("Total shots") == 3
    assert "first" in output
    assert "second" in output
    assert "Suite total" in output
    assert "2/2" in output
    assert output.count("benchmark chatter") == 2
    assert output.count("benchmark setup chatter") == 2
    assert "(min)" not in output


def test_suite_estimate_all_guarded_entries_prints_only_total_by_default(
    tmp_path, estimation, monkeypatch, capsys
):
    monkeypatch.setattr(run, "print_resource_estimate", print_resource_estimate)

    run.estimate_suite(_args(_write_suite(tmp_path, guarded=True), all_components=True))

    assert estimation.handler.estimate_resources_handler.call_count == 2
    assert capsys.readouterr().out == (
        "Cost estimate for Suite total (test_suite) on ibm:ibm_phoenix\n"
        "Estimated IBM QPU time (s): 25.000\n"
    )


def test_suite_ibm_timing_failure_keeps_resources_and_marks_usage_total_partial(
    tmp_path, estimation, monkeypatch, capsys
):
    printer = MagicMock(wraps=print_resource_estimate)
    monkeypatch.setattr(run, "print_resource_estimate", printer)
    estimation.ibm_usage.side_effect = [12.5, ValueError("Missing measurement duration")]

    run.estimate_suite(_args(_write_suite(tmp_path), verbose=True))

    first, failed, total = [call.args[3] for call in printer.call_args_list]
    assert first.ibm_quantum_seconds == 12.5
    assert first.ibm_timing_complete
    assert failed.ibm_quantum_seconds is None
    assert not failed.ibm_timing_complete
    assert failed.circuit_count == 1
    assert failed.total_shots == 10
    assert total.circuit_count == 2
    assert total.total_shots == 20
    assert total.ibm_quantum_seconds == 12.5
    assert not total.ibm_timing_complete
    output = capsys.readouterr().out
    assert "Missing measurement duration" in output
    assert "Partial IBM QPU time (s)" in output
    assert "(min)" not in output
    assert "HQC" not in output


def test_suite_estimate_partial_time_is_concise_by_default(
    tmp_path, estimation, monkeypatch, capsys
):
    monkeypatch.setattr(run, "print_resource_estimate", print_resource_estimate)
    estimation.ibm_usage.side_effect = [12.5, ValueError("Missing measurement duration")]

    run.estimate_suite(_args(_write_suite(tmp_path)))

    assert capsys.readouterr().out.strip() == (
        "Cost estimate for Suite total (test_suite) on ibm:ibm_phoenix\n"
        "Partial IBM QPU time (s): 12.500 (use --verbose for details)"
    )


def test_suite_estimate_without_device_is_concise_by_default(
    tmp_path, estimation, monkeypatch, capsys
):
    monkeypatch.setattr(run, "print_resource_estimate", print_resource_estimate)

    run.estimate_suite(_args(_write_suite(tmp_path), device=None))

    estimation.setup_device.assert_not_called()
    assert capsys.readouterr().out.strip() == (
        "Cost estimate for Suite total (test_suite) on ibm:(no device)\n"
        "Estimated IBM QPU time (s): n/a (use --verbose for details)"
    )


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize(
    ("components", "second_component", "expected_label"),
    [
        (None, "wit", "Partial suite total (test_suite)"),
        ("QFT", "qft", "Partial component total (test_suite: qft)"),
        (
            ["WIT", "QFT"],
            "wit",
            "Partial selected components total (test_suite: qft, wit)",
        ),
    ],
)
def test_suite_estimate_marks_total_partial_when_entry_cannot_be_built(
    tmp_path, estimation, monkeypatch, capsys, verbose, components, second_component, expected_label
):
    monkeypatch.setattr(run, "print_resource_estimate", print_resource_estimate)
    estimation.handler.estimate_resources_handler.side_effect = [
        [CircuitBatch(circuits=[QuantumCircuit(6)], shots=10)],
        NotImplementedError("Estimation is not implemented"),
    ]

    path = _write_suite(
        tmp_path,
        [_entry("first", component="qft"), _entry("second", component=second_component)],
    )
    run.estimate_suite(_args(path, components=components, verbose=verbose))

    output = capsys.readouterr().out
    assert f"Cost estimate for {expected_label} on ibm:ibm_phoenix" in output
    assert "second: NotImplementedError: Estimation is not implemented" in output
    if verbose:
        assert "Partial IBM QPU time (s)" in output
        assert "Total shots" in output
    else:
        assert "Partial IBM QPU time (s): 12.500 (use --verbose for details)" in output
        assert "Estimated IBM QPU time" not in output
        assert "Total shots" not in output


def test_suite_estimate_requires_provider(tmp_path, estimation, capsys):
    run.estimate_suite(_args(_write_suite(tmp_path), provider=None))

    assert "Provider is required" in capsys.readouterr().out
    estimation.setup_device.assert_not_called()
    estimation.setup_benchmark.assert_not_called()
    estimation.printer.assert_not_called()


@pytest.mark.parametrize(
    ("components", "all_components", "message"),
    [
        (None, False, "--all"),
        (["missing"], False, "Unknown component(s): missing"),
        (["qft", "QFT"], False, "Duplicate component selection"),
        (["qft"], True, "--component and --all cannot be used together"),
    ],
)
def test_suite_estimate_rejects_invalid_selection_before_device_setup(
    tmp_path, estimation, capsys, components, all_components, message
):
    path = _write_suite(tmp_path, [_entry("qft", component="qft")], guarded=True)

    run.estimate_suite(_args(path, components=components, all_components=all_components))

    assert message in capsys.readouterr().out
    estimation.setup_device.assert_not_called()
    estimation.setup_benchmark.assert_not_called()
    estimation.printer.assert_not_called()


@pytest.mark.parametrize("components", [["QFT"], "qft"])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("middle_component", ["wit", "qft"])
def test_suite_estimate_selects_component_entries_in_suite_order(
    tmp_path, estimation, monkeypatch, capsys, components, verbose, middle_component
):
    printer = MagicMock(wraps=print_resource_estimate)
    monkeypatch.setattr(run, "print_resource_estimate", printer)
    path = _write_suite(
        tmp_path,
        [
            _entry("qft_small", component="qft", shots=10),
            _entry("middle", component=middle_component, shots=20),
            _entry("qft_large", component="qft", shots=30),
        ],
        guarded=True,
    )

    run.estimate_suite(_args(path, components=components, verbose=verbose))

    expected_shots = [10, 30] if middle_component == "wit" else [10, 20, 30]
    entry_count = len(expected_shots)
    assert [
        call.args[1].shots for call in estimation.setup_benchmark.call_args_list
    ] == expected_shots
    assert printer.call_count == (entry_count + 1 if verbose else 1)
    assert printer.call_args.args[0] == "Component total (test_suite: qft)"
    output = capsys.readouterr().out
    assert "Cost estimate for Component total (test_suite: qft) on ibm:ibm_phoenix" in output
    assert "Suite total" not in output
    if verbose:
        assert "qft_small" in output
        assert "qft_large" in output
        assert f"{entry_count}/{entry_count}" in output
    else:
        assert len(output.splitlines()) == 2


@pytest.mark.parametrize(
    ("components", "all_components", "entry_count", "expected_label"),
    [
        (None, True, 22, "Suite total (metriq_score_1_0)"),
        (
            ["QFT", "wit"],
            False,
            5,
            "Selected components total (metriq_score_1_0: wit, qft)",
        ),
    ],
)
@pytest.mark.parametrize("verbose", [False, True])
def test_suite_estimate_supports_bundled_suite_and_repeatable_components(
    estimation,
    monkeypatch,
    capsys,
    components,
    all_components,
    entry_count,
    expected_label,
    verbose,
):
    printer = MagicMock(wraps=print_resource_estimate)
    monkeypatch.setattr(run, "print_resource_estimate", printer)
    suite = parse_suite_file("metriq_score_1_0")
    expected_entries = suite.select_components(components) if components else suite.benchmarks

    run.estimate_suite(
        _args(
            "metriq_score_1_0",
            components=components,
            all_components=all_components,
            verbose=verbose,
        )
    )

    estimation.setup_device.assert_called_once_with("ibm", "ibm_phoenix")
    assert estimation.setup_benchmark.call_count == entry_count
    assert [call.args[1].benchmark_name for call in estimation.setup_benchmark.call_args_list] == [
        entry.config["benchmark_name"] for entry in expected_entries
    ]
    assert printer.call_count == (entry_count + 1 if verbose else 1)
    assert printer.call_args.args[0] == expected_label
    output = capsys.readouterr().out
    assert f"Cost estimate for {expected_label} on ibm:ibm_phoenix" in output
    if verbose:
        assert f"{entry_count}/{entry_count}" in output
    else:
        assert len(output.splitlines()) == 2


def test_suite_estimate_continues_after_invalid_config_capacity_and_handler_failures(
    tmp_path, estimation, capsys
):
    estimation.device.num_qubits = 6
    path = _write_suite(
        tmp_path,
        [
            _entry("invalid_config", shots=0),
            _entry("too_many_qubits", num_qubits=7),
            _entry("unsupported_estimator"),
            _entry("unexpected_error"),
            _entry("successful_entry"),
        ],
    )
    estimation.handler.estimate_resources_handler.side_effect = [
        NotImplementedError("Estimation is not implemented"),
        RuntimeError("Circuit construction failed"),
        [CircuitBatch(circuits=[QuantumCircuit(6)], shots=25)],
    ]

    run.estimate_suite(_args(path, verbose=True))

    assert estimation.setup_benchmark.call_count == 3
    assert estimation.printer.call_count == 2
    label, _, _, total = estimation.printer.call_args.args
    assert "Partial suite total" in label
    assert total.job_count == 1
    assert total.circuit_count == 1
    assert total.total_shots == 25
    output = capsys.readouterr().out
    for message in (
        "invalid_config",
        "too_many_qubits",
        "Requested 7 qubits",
        "supports only 6",
        "Estimation is not implemented",
        "Circuit construction failed",
        "successful_entry",
        "1/5",
    ):
        assert message in output


def test_suite_estimate_prints_no_total_when_every_estimator_fails(tmp_path, estimation, capsys):
    estimation.handler.estimate_resources_handler.side_effect = ValueError("Device is required")

    run.estimate_suite(_args(_write_suite(tmp_path), device=None, verbose=True))

    assert estimation.handler.estimate_resources_handler.call_count == 2
    estimation.printer.assert_not_called()
    output = capsys.readouterr().out
    assert "No benchmarks could be estimated." in output
    assert "0/2" in output


@pytest.mark.parametrize(
    ("contents", "message"),
    [("{invalid json}", "Invalid suite configuration"), (None, "not found")],
)
def test_suite_estimate_handles_invalid_or_missing_suite(
    tmp_path, estimation, capsys, contents, message
):
    path = tmp_path / "invalid.json"
    if contents is not None:
        path.write_text(contents)

    run.estimate_suite(_args(str(path)))

    assert message in capsys.readouterr().out
    estimation.setup_device.assert_not_called()
    estimation.setup_benchmark.assert_not_called()
    estimation.printer.assert_not_called()


def test_suite_estimate_handles_empty_suite(tmp_path, estimation, capsys):
    run.estimate_suite(_args(_write_suite(tmp_path, [])))

    assert "No benchmarks found" in capsys.readouterr().out
    estimation.setup_device.assert_not_called()
    estimation.printer.assert_not_called()
