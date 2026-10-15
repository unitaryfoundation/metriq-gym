"""Unit tests for CLOPS benchmark circuit construction and dispatch modes."""

import argparse
from unittest.mock import MagicMock, patch

import pytest
from qbraid.runtime import QiskitBackend
import rustworkx as rx
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.clops import (
    Clops,
    ClopsData,
    ClopsResult,
    append_1q_layer,
    create_qubit_list,
    instantiate_circuits,
    prepare_clops_template,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_graph(n: int) -> rx.PyGraph:
    """Return a linear graph 0-1-2-..-(n-1)."""
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    for i in range(n - 1):
        g.add_edge(i, i + 1, None)
    return g


def _make_params(**overrides):
    """Build a mock params object with CLOPS defaults."""
    defaults = dict(
        benchmark_name="CLOPS",
        num_qubits=4,
        num_layers=2,
        num_circuits=3,
        shots=10,
        seed=42,
        two_qubit_gate="cz",
        mode="instantiated",
        use_session=False,
    )
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _make_clops(**param_overrides) -> Clops:
    args = argparse.Namespace()
    params = _make_params(**param_overrides)
    return Clops(args, params)


# ---------------------------------------------------------------------------
# create_qubit_list
# ---------------------------------------------------------------------------


def test_create_qubit_list_linear():
    g = _linear_graph(6)
    qubits = create_qubit_list(4, g)
    assert len(qubits) == 4
    # all qubits should be valid node indices
    assert all(q in g.node_indices() for q in qubits)


def test_create_qubit_list_too_few_qubits():
    g = _linear_graph(3)
    with pytest.raises(ValueError, match="cannot create set of 5"):
        create_qubit_list(5, g)


# ---------------------------------------------------------------------------
# append_1q_layer
# ---------------------------------------------------------------------------


def test_append_1q_layer_parameterized():
    qc = QuantumCircuit(4)
    qubits = list(qc.qubits)
    params = append_1q_layer(qc, qubits, parameterized=True, parameter_prefix="T")
    assert len(params) == 3  # 3 ParameterVectors (pars0, pars1, pars2)
    assert len(qc.parameters) == 3 * 4  # 3 vectors × 4 qubits


def test_append_1q_layer_fixed():
    qc = QuantumCircuit(4)
    qubits = list(qc.qubits)
    append_1q_layer(qc, qubits, parameterized=False)
    # Still returns parameter vectors but they aren't in the circuit
    assert len(qc.parameters) == 0


# ---------------------------------------------------------------------------
# prepare_clops_template
# ---------------------------------------------------------------------------


def test_prepare_clops_template_parameterized():
    g = _linear_graph(6)
    template, params = prepare_clops_template(
        width=4, layers=2, two_qubit_gate="cz", topology_graph=g, parameterized=True, seed=0
    )
    assert isinstance(template, QuantumCircuit)
    # Should have parameters
    assert len(template.parameters) > 0
    # params is a flat list of ParameterVectors
    assert len(params) == 2 * 3  # 2 layers × 3 vectors per layer


def test_prepare_clops_template_fixed():
    g = _linear_graph(6)
    template, params = prepare_clops_template(
        width=4, layers=2, two_qubit_gate="cz", topology_graph=g, parameterized=False, seed=0
    )
    assert isinstance(template, QuantumCircuit)
    assert len(template.parameters) == 0


# ---------------------------------------------------------------------------
# instantiate_circuits
# ---------------------------------------------------------------------------


def test_instantiate_circuits():
    g = _linear_graph(6)
    template, params = prepare_clops_template(
        width=4, layers=2, two_qubit_gate="cz", topology_graph=g, parameterized=True, seed=0
    )
    circuits = instantiate_circuits(template, params, num_circuits=5, seed=99)
    assert len(circuits) == 5
    # Each circuit should have no remaining parameters
    for circ in circuits:
        assert len(circ.parameters) == 0


# ---------------------------------------------------------------------------
# Dispatch: instantiated mode
# ---------------------------------------------------------------------------


def test_dispatch_instantiated_calls_device_run():
    clops = _make_clops(mode="instantiated", use_session=False)
    device = MagicMock()
    device.profile.basis_gates = ["cz", "sx", "rz"]

    # Provide a simple graph for connectivity_graph
    graph = _linear_graph(6)
    mock_job = MagicMock()
    mock_job.id = "job-123"
    device.run.return_value = mock_job

    with (
        patch("metriq_gym.benchmarks.clops.connectivity_graph_for_gate", return_value=graph),
        patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph),
        patch("metriq_gym.benchmarks.clops.pruned_connectivity_graph", return_value=graph),
    ):
        result = clops.dispatch_handler(device)

    assert isinstance(result, ClopsData)
    device.run.assert_called_once()
    # Check circuits were passed
    circuits_arg = device.run.call_args[0][0]
    assert len(circuits_arg) == 3  # num_circuits=3
    assert all(isinstance(c, QuantumCircuit) for c in circuits_arg)


# ---------------------------------------------------------------------------
# Dispatch: parameterized mode requires IBM device
# ---------------------------------------------------------------------------


def test_dispatch_parameterized_rejects_non_ibm_device():
    clops = _make_clops(mode="parameterized")
    device = MagicMock()  # not an IBM device

    with pytest.raises(ValueError, match="requires the ibm provider"):
        clops.dispatch_handler(device)


def test_dispatch_parameterized_calls_submit_with_pub():
    clops = _make_clops(mode="parameterized")
    device = MagicMock(spec=QiskitBackend)
    device.profile.basis_gates = ["cz", "sx", "rz"]

    graph = _linear_graph(6)
    mock_job = MagicMock()
    mock_job.id = "job-456"
    clops._submit_ibm_with_options = MagicMock(return_value=mock_job)

    with (
        patch("metriq_gym.benchmarks.clops.connectivity_graph_for_gate", return_value=graph),
        patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph),
        patch("metriq_gym.benchmarks.clops.pruned_connectivity_graph", return_value=graph),
    ):
        result = clops.dispatch_handler(device)

    assert isinstance(result, ClopsData)
    clops._submit_ibm_with_options.assert_called_once()
    kwargs = clops._submit_ibm_with_options.call_args
    pubs = kwargs.kwargs["pubs"]
    assert len(pubs) == 1
    pub = pubs[0]
    # PUB is (circuit, param_values, shots)
    assert isinstance(pub[0], QuantumCircuit)
    assert len(pub[1]) == 3  # num_circuits=3 sets of parameters
    assert pub[2] == 10  # shots


# ---------------------------------------------------------------------------
# Dispatch: twirled mode requires IBM Device
# ---------------------------------------------------------------------------


def test_dispatch_twirled_rejects_non_ibm_device():
    clops = _make_clops(mode="twirled")
    device = MagicMock()

    with pytest.raises(ValueError, match="requires the ibm provider"):
        clops.dispatch_handler(device)


def test_dispatch_twirled_calls_submit_with_twirling_opts():
    from qiskit_ibm_runtime.options import TwirlingOptions

    clops = _make_clops(mode="twirled")
    device = MagicMock(spec=QiskitBackend)
    device.profile.basis_gates = ["cz", "sx", "rz"]

    graph = _linear_graph(6)
    mock_job = MagicMock()
    mock_job.id = "job-789"
    clops._submit_ibm_with_options = MagicMock(return_value=mock_job)

    with (
        patch("metriq_gym.benchmarks.clops.connectivity_graph_for_gate", return_value=graph),
        patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph),
        patch("metriq_gym.benchmarks.clops.pruned_connectivity_graph", return_value=graph),
    ):
        result = clops.dispatch_handler(device)

    assert isinstance(result, ClopsData)
    clops._submit_ibm_with_options.assert_called_once()
    kwargs = clops._submit_ibm_with_options.call_args.kwargs
    # Should have a single fixed circuit (no parameters)
    pubs = kwargs["pubs"]
    assert len(pubs) == 1
    assert isinstance(pubs[0], QuantumCircuit)
    assert len(pubs[0].parameters) == 0

    # Should pass twirling options
    tw = kwargs["twirling_options"]
    assert isinstance(tw, TwirlingOptions)
    assert tw.num_randomizations == 3  # num_circuits
    assert tw.shots_per_randomization == 10  # shots
    assert tw.enable_gates is True

    # Total shots = shots * num_circuits
    assert kwargs["shots"] == 30


# ---------------------------------------------------------------------------
# Dispatch: unknown mode
# ---------------------------------------------------------------------------


def test_dispatch_unknown_mode_raises():
    clops = _make_clops(mode="bogus", use_session=False)
    device = MagicMock()
    with pytest.raises(ValueError, match="Unknown CLOPS mode"):
        clops.dispatch_handler(device)


# ---------------------------------------------------------------------------
# Static resource estimation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "use_session", "transpile"),
    [
        ("instantiated", False, True),
        ("instantiated", True, False),
        ("parameterized", False, False),
        ("parameterized", True, False),
    ],
)
def test_estimate_ibm_binds_submitted_parameters_without_execution(mode, use_session, transpile):
    clops = _make_clops(mode=mode, use_session=use_session)
    device = MagicMock(spec=QiskitBackend)
    template, parameters = prepare_clops_template(
        width=4, layers=2, two_qubit_gate="cz", topology_graph=_linear_graph(6), seed=42
    )
    expected = instantiate_circuits(template, parameters, num_circuits=3, seed=42)

    with (
        patch.object(clops, "_build_template", return_value=(template, parameters)) as build,
        patch.object(clops, "_submit_ibm_with_options") as submit,
        patch("qiskit_ibm_runtime.Session") as session,
    ):
        batches = clops.estimate_resources_handler(device)

    assert len(batches) == 1
    batch = batches[0]
    assert batch.circuits == expected
    assert all(not circuit.parameters for circuit in batch.circuits)
    assert len(batch.circuits) * batch.shots == 30
    assert batch.transpile is transpile
    assert batch.notes == (
        ["Session idle time is excluded; billed session usage can be higher."]
        if use_session
        else []
    )
    build.assert_called_once_with(device, parameterized=True)
    device.run.assert_not_called()
    submit.assert_not_called()
    session.assert_not_called()


def test_estimate_twirled_reuses_template_and_counts_all_randomizations():
    clops = _make_clops(mode="twirled", num_circuits=1000, shots=100, use_session=True)
    device = MagicMock(spec=QiskitBackend)
    template, parameters = prepare_clops_template(
        width=4,
        layers=2,
        two_qubit_gate="cz",
        topology_graph=_linear_graph(6),
        parameterized=False,
    )

    with (
        patch.object(clops, "_build_template", return_value=(template, parameters)) as build,
        patch.object(clops, "_submit_ibm_with_options") as submit,
        patch("qiskit_ibm_runtime.Session") as session,
        patch("metriq_gym.benchmarks.clops.instantiate_circuits") as instantiate,
    ):
        batches = clops.estimate_resources_handler(device)

    assert len(batches) == 1
    batch = batches[0]
    assert len(batch.circuits) == 1000
    assert all(circuit is template for circuit in batch.circuits)
    assert len(batch.circuits) * batch.shots == 100_000
    assert batch.transpile is False
    assert batch.notes == [
        "Twirled CLOPS uses the untwirled template; runtime twirling overhead is excluded.",
        "Session idle time is excluded; billed session usage can be higher.",
    ]
    build.assert_called_once_with(device, parameterized=False)
    device.run.assert_not_called()
    submit.assert_not_called()
    session.assert_not_called()
    instantiate.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "use_session"),
    [("parameterized", False), ("twirled", False), ("instantiated", True)],
)
def test_estimate_ibm_only_options_reject_non_ibm_device(mode, use_session):
    clops = _make_clops(mode=mode, use_session=use_session)
    device = MagicMock()

    with (
        patch.object(clops, "_build_template") as build,
        pytest.raises(ValueError, match="requires the ibm provider"),
    ):
        clops.estimate_resources_handler(device)

    build.assert_not_called()
    device.run.assert_not_called()


def test_estimate_non_ibm_instantiated_transpiles_normally():
    clops = _make_clops()
    device = MagicMock()
    template, parameters = prepare_clops_template(
        width=4, layers=2, two_qubit_gate="cz", topology_graph=_linear_graph(6)
    )
    with patch.object(clops, "_build_template", return_value=(template, parameters)):
        batch = clops.estimate_resources_handler(device)[0]

    assert batch.transpile is True
    assert batch.notes == []
    assert len(batch.circuits) * batch.shots == 30
    device.run.assert_not_called()


def test_estimate_unknown_mode_raises():
    clops = _make_clops(mode="bogus")
    with pytest.raises(ValueError, match="Unknown CLOPS mode"):
        clops.estimate_resources_handler(MagicMock())


# ---------------------------------------------------------------------------
# ClopsResult
# ---------------------------------------------------------------------------


def test_clops_result_score():
    r = ClopsResult(clops_score=42.5)
    assert r.compute_score().value == pytest.approx(42.5)


def test_clops_result_steady_state_defaults_to_none():
    r = ClopsResult(clops_score=100.0)
    assert r.steady_state_clops is None


def test_clops_result_steady_state_included_when_set():
    r = ClopsResult(clops_score=100.0, steady_state_clops=200.0)
    assert r.steady_state_clops == pytest.approx(200.0)
    # steady_state_clops should appear in values
    assert "steady_state_clops" in r.values


# ---------------------------------------------------------------------------
# _compute_steady_state_clops
# ---------------------------------------------------------------------------


def test_steady_state_clops_with_multiple_spans():
    """Verify the steady-state formula with two mock execution spans."""
    from datetime import datetime, timedelta, timezone
    from metriq_gym.benchmarks.clops import _compute_steady_state_clops
    from qbraid.runtime import QiskitJob

    # Build mock spans: span0 (size=100, 0s-1s), span1 (size=400, 1s-3s)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    span0 = MagicMock()
    span0.size = 100
    span0.start = t0
    span0.stop = t0 + timedelta(seconds=1)

    span1 = MagicMock()
    span1.size = 400
    span1.start = t0 + timedelta(seconds=1)
    span1.stop = t0 + timedelta(seconds=3)

    sorted_spans = MagicMock()
    sorted_spans.__iter__ = MagicMock(return_value=iter([span0, span1]))
    sorted_spans.__len__ = MagicMock(return_value=2)
    sorted_spans.__getitem__ = MagicMock(side_effect=lambda i: [span0, span1][i])
    sorted_spans.stop = t0 + timedelta(seconds=3)  # last span stop

    execution_spans = MagicMock()
    execution_spans.sort.return_value = sorted_spans

    mock_result = MagicMock()
    mock_result.metadata = {"execution": {"execution_spans": execution_spans}}

    mock_runtime_job = MagicMock()
    mock_runtime_job.result.return_value = mock_result

    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job._job = mock_runtime_job
    qiskit_job.id = "test-job"

    num_layers = 10
    # Formula: ((500 - 100) * 10) / (3s - 1s) = 4000 / 2 = 2000
    result = _compute_steady_state_clops([qiskit_job], num_layers)
    assert result == 2000


def test_steady_state_clops_single_span_returns_none():
    """With only one span, can't exclude startup — should return None."""
    from datetime import datetime, timedelta, timezone
    from metriq_gym.benchmarks.clops import _compute_steady_state_clops
    from qbraid.runtime import QiskitJob

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    span0 = MagicMock()
    span0.size = 100
    span0.stop = t0 + timedelta(seconds=1)

    sorted_spans = MagicMock()
    sorted_spans.__len__ = MagicMock(return_value=1)

    execution_spans = MagicMock()
    execution_spans.sort.return_value = sorted_spans

    mock_result = MagicMock()
    mock_result.metadata = {"execution": {"execution_spans": execution_spans}}

    mock_runtime_job = MagicMock()
    mock_runtime_job.result.return_value = mock_result

    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job._job = mock_runtime_job
    qiskit_job.id = "test-job"

    assert _compute_steady_state_clops([qiskit_job], num_layers=10) is None


def test_steady_state_clops_non_qiskit_job_returns_none():
    """Non-QiskitJob should be skipped, returning None."""
    from metriq_gym.benchmarks.clops import _compute_steady_state_clops

    generic_job = MagicMock()  # not a QiskitJob
    assert _compute_steady_state_clops([generic_job], num_layers=10) is None


def test_steady_state_clops_missing_metadata_returns_none():
    """If metadata is missing execution_spans, should return None gracefully."""
    from metriq_gym.benchmarks.clops import _compute_steady_state_clops
    from qbraid.runtime import QiskitJob

    mock_result = MagicMock()
    mock_result.metadata = {}  # no "execution" key

    mock_runtime_job = MagicMock()
    mock_runtime_job.result.return_value = mock_result

    qiskit_job = MagicMock(spec=QiskitJob)
    qiskit_job._job = mock_runtime_job
    qiskit_job.id = "test-job"

    assert _compute_steady_state_clops([qiskit_job], num_layers=10) is None
