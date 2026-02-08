"""Unit tests for CLOPS benchmark circuit construction and dispatch modes."""

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rustworkx as rx
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.clops import (
    Clops,
    ClopsData,
    ClopsResult,
    append_1q_layer,
    append_2q_layer,
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
    params = append_1q_layer(qc, qubits, parameterized=False)
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
    clops = _make_clops(mode="instantiated")
    device = MagicMock()
    device.profile.basis_gates = ["cz", "sx", "rz"]

    # Provide a simple graph for connectivity_graph
    graph = _linear_graph(6)
    mock_job = MagicMock()
    mock_job.id = "job-123"
    device.run.return_value = mock_job

    with patch("metriq_gym.benchmarks.clops.connectivity_graph_for_gate", return_value=graph), \
         patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph), \
         patch("metriq_gym.benchmarks.clops.pruned_connectivity_graph", return_value=graph):
        result = clops.dispatch_handler(device)

    assert isinstance(result, ClopsData)
    device.run.assert_called_once()
    # Check circuits were passed
    circuits_arg = device.run.call_args[0][0]
    assert len(circuits_arg) == 3  # num_circuits=3
    assert all(isinstance(c, QuantumCircuit) for c in circuits_arg)


# ---------------------------------------------------------------------------
# Dispatch: parameterized mode requires IBMSamplerDevice
# ---------------------------------------------------------------------------

def test_dispatch_parameterized_rejects_non_ibm_device():
    clops = _make_clops(mode="parameterized")
    device = MagicMock()  # not an IBMSamplerDevice

    with pytest.raises(ValueError, match="requires the ibm_sampler provider"):
        clops.dispatch_handler(device)


def test_dispatch_parameterized_calls_submit_with_pub():
    from metriq_gym.ibm_sampler.device import IBMSamplerDevice

    clops = _make_clops(mode="parameterized")
    device = MagicMock(spec=IBMSamplerDevice)
    device.profile.basis_gates = ["cz", "sx", "rz"]

    graph = _linear_graph(6)
    mock_job = MagicMock()
    mock_job.id = "job-456"
    device.submit.return_value = mock_job

    with patch("metriq_gym.benchmarks.clops.connectivity_graph_for_gate", return_value=graph), \
         patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph), \
         patch("metriq_gym.benchmarks.clops.pruned_connectivity_graph", return_value=graph):
        result = clops.dispatch_handler(device)

    assert isinstance(result, ClopsData)
    device.submit.assert_called_once()
    kwargs = device.submit.call_args
    pubs = kwargs.kwargs["pubs"]
    assert len(pubs) == 1
    pub = pubs[0]
    # PUB is (circuit, param_values, shots)
    assert isinstance(pub[0], QuantumCircuit)
    assert len(pub[1]) == 3  # num_circuits=3 sets of parameters
    assert pub[2] == 10  # shots


# ---------------------------------------------------------------------------
# Dispatch: twirled mode requires IBMSamplerDevice
# ---------------------------------------------------------------------------

def test_dispatch_twirled_rejects_non_ibm_device():
    clops = _make_clops(mode="twirled")
    device = MagicMock()

    with pytest.raises(ValueError, match="requires the ibm_sampler provider"):
        clops.dispatch_handler(device)


def test_dispatch_twirled_calls_submit_with_twirling_opts():
    from metriq_gym.ibm_sampler.device import IBMSamplerDevice
    from qiskit_ibm_runtime.options import TwirlingOptions

    clops = _make_clops(mode="twirled")
    device = MagicMock(spec=IBMSamplerDevice)
    device.profile.basis_gates = ["cz", "sx", "rz"]

    graph = _linear_graph(6)
    mock_job = MagicMock()
    mock_job.id = "job-789"
    device.submit.return_value = mock_job

    with patch("metriq_gym.benchmarks.clops.connectivity_graph_for_gate", return_value=graph), \
         patch("metriq_gym.benchmarks.clops.connectivity_graph", return_value=graph), \
         patch("metriq_gym.benchmarks.clops.pruned_connectivity_graph", return_value=graph):
        result = clops.dispatch_handler(device)

    assert isinstance(result, ClopsData)
    device.submit.assert_called_once()
    kwargs = device.submit.call_args.kwargs
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
    clops = _make_clops(mode="bogus")
    device = MagicMock()
    with pytest.raises(ValueError, match="Unknown CLOPS mode"):
        clops.dispatch_handler(device)


# ---------------------------------------------------------------------------
# ClopsResult
# ---------------------------------------------------------------------------

def test_clops_result_score():
    r = ClopsResult(clops_score=42.5)
    assert r.compute_score().value == pytest.approx(42.5)
