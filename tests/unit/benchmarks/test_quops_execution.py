"""QUOPS compilation boundaries, using local SDKs and mocked submissions only."""

from types import SimpleNamespace
from unittest.mock import Mock

import networkx as nx
import numpy as np
import pytest
from pytket.circuit import Circuit, OpType
from pytket.architecture import FullyConnected
from qbraid.programs import ExperimentType, ProgramSpec
from qbraid.runtime import BraketDevice, QiskitBackend, QiskitJob, TargetProfile
from qbraid.runtime.quantinuum import QuantinuumDevice
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import InstructionProperties
from qiskit_aer import AerSimulator

from metriq_gym.benchmarks.quops.circuits import build_mirror_circuits, build_spam_circuit
from metriq_gym.local.device import LocalAerDevice
from metriq_gym.qplatform.job import qpu_time
from metriq_gym.benchmarks.quops.execution import (
    _braket_circuit,
    compile_quops_circuit,
    prepare_quops_circuit,
    submit_quops_circuits,
    validate_quops_options,
)


def _backend():
    return GenericBackendV2(
        5,
        coupling_map=[[i, i + 1] for i in range(4)] + [[i + 1, i] for i in range(4)],
        basis_gates=["rz", "sx", "x", "cz"],
        seed=42,
    )


def _ibm(backend=None):
    backend = _backend() if backend is None else backend
    profile = TargetProfile(
        device_id=backend.name,
        simulator=False,
        num_qubits=backend.num_qubits,
        experiment_type=ExperimentType.GATE_MODEL,
        program_spec=ProgramSpec(QuantumCircuit),
    )
    return QiskitBackend(profile, service=SimpleNamespace(backend=lambda *a, **kw: backend))


def _quantinuum():
    profile = TargetProfile(
        device_id="H2-1",
        simulator=False,
        num_qubits=8,
        experiment_type=ExperimentType.GATE_MODEL,
        program_spec=ProgramSpec(Circuit, alias="pytket"),
        provider_name="quantinuum",
        backend_info=SimpleNamespace(architecture=FullyConnected(8)),
    )
    return QuantinuumDevice(profile)


def _workload():
    circuit = QuantumCircuit(3)
    circuit.rx(0.21, 0)
    circuit.ry(1.37, 1)
    circuit.rz(-0.63, 2)
    circuit.cx(0, 2)
    circuit.cx(1, 0)
    return circuit


def _mirror(circuit):
    mirrored = circuit.copy()
    mirrored.barrier()
    mirrored.compose(circuit.inverse(), inplace=True)
    mirrored.measure_all()
    return mirrored


def _assert_zero_state(circuit):
    probabilities = Statevector.from_instruction(
        circuit.remove_final_measurements(inplace=False)
    ).probabilities()
    assert probabilities[0] == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("local", [False, True], ids=["ibm", "noisy-local"])
def test_native_fixed_subset_compilation_and_mirror_preparation(local):
    device = (
        LocalAerDevice(provider=Mock(), backend=AerSimulator.from_backend(_backend()))
        if local
        else _ibm()
    )
    compiled = compile_quops_circuit(device, _workload(), physical_qubits=[3, 1, 2], seed=8)
    assert compiled.physical_qubits == [1, 2, 3]
    assert compiled.circuit.num_qubits == 3
    assert set(compiled.circuit.count_ops()) <= {"u", "cx", "cz"}
    prepared = prepare_quops_circuit(device, _mirror(compiled.circuit), compiled.physical_qubits)
    used = {
        prepared.find_bit(q).index
        for inst in prepared.data
        if inst.operation.name != "barrier"
        for q in inst.qubits
    }
    assert used == {1, 2, 3}
    assert prepared.count_ops()["cz"] == 2 * compiled.circuit.count_ops()["cz"]
    assert prepared.count_ops()["barrier"] >= 2
    _assert_zero_state(prepared)


def test_global_ibm_layout_has_no_active_ancillas():
    device = _ibm()
    compiled = compile_quops_circuit(device, _workload(), seed=9)
    assert len(compiled.physical_qubits) == compiled.circuit.num_qubits == 3
    assert set(compiled.physical_qubits) <= set(range(5))
    prepared = prepare_quops_circuit(device, _mirror(compiled.circuit), compiled.physical_qubits)
    _assert_zero_state(prepared)


@pytest.mark.parametrize("healthy_qubits", [[0, 1, 2], [3, 4, 5]])
def test_global_ibm_layout_uses_backend_calibrations(healthy_qubits):
    # Either end of this chain can run the workload with the same routing cost.
    # Change only the calibration errors to make one region preferable.
    backend = GenericBackendV2(
        6,
        coupling_map=[[a, b] for i in range(5) for a, b in ((i, i + 1), (i + 1, i))],
        basis_gates=["rz", "sx", "x", "cz"],
        seed=42,
    )
    for name in backend.target.operation_names:
        for qargs, properties in list(backend.target[name].items()):
            if properties is None:
                continue
            error = 0.001 if all(q in healthy_qubits for q in qargs) else 0.2
            backend.target.update_instruction_properties(
                name,
                qargs,
                InstructionProperties(
                    duration=properties.duration, error=0.0 if name == "rz" else error
                ),
            )

    compiled = compile_quops_circuit(_ibm(backend), _workload(), seed=1)

    assert compiled.physical_qubits == healthy_qubits


def test_bundled_cz_backend_compiles_full_sized_workload_without_provider_calls():
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    from metriq_gym.local.provider import _LazyAerBackend
    from metriq_gym.benchmarks.quops.circuits import sample_quops_circuit, build_mirror_circuits

    backend = FakeMarrakesh()
    device = LocalAerDevice(provider=None, backend=_LazyAerBackend(backend))
    circuit = sample_quops_circuit(6, 216, np.random.default_rng(42))
    layout = compile_quops_circuit(device, circuit, seed=42).physical_qubits
    compiled = compile_quops_circuit(device, circuit, seed=43, physical_qubits=layout)
    mirror = build_mirror_circuits(compiled.circuit, np.random.default_rng(44))[0]
    prepared = prepare_quops_circuit(device, mirror.circuit, layout)
    assert len(layout) == 6
    assert prepared.count_ops()["cz"] == 2 * compiled.circuit.count_ops()["cz"]
    used = {
        prepared.find_bit(q).index
        for inst in prepared.data
        if inst.operation.name != "barrier"
        for q in inst.qubits
    }
    assert used == set(layout)
    assert compiled.metadata["native_gates"] == ["cz", "rz", "sx", "x"]
    assert "qiskit-aer" in compiled.metadata["sdk_versions"]


def test_global_routing_ancillas_fall_back_to_connected_subset(monkeypatch):
    from qiskit import transpile

    device = _ibm()
    with_ancilla = QuantumCircuit(5)
    with_ancilla.cx(0, 4)
    with_ancilla._layout = SimpleNamespace(
        initial_index_layout=lambda **kw: [0, 1, 2],
        final_index_layout=lambda **kw: [0, 1, 4],
    )

    def transpile_with_ancilla(*args, **kwargs):
        return with_ancilla if "backend" in kwargs else transpile(*args, **kwargs)

    monkeypatch.setattr("metriq_gym.benchmarks.quops.execution.transpile", transpile_with_ancilla)
    compiled = compile_quops_circuit(device, _workload(), seed=9)
    assert compiled.metadata["layout_method"] == "connected_subset_fallback"
    assert compiled.physical_qubits == [0, 1, 2]
    assert compiled.circuit.num_qubits == 3


def test_ibm_submission_disables_automatic_twirling_and_enables_requested_dd(monkeypatch):
    device = _ibm()
    sampler = Mock()
    constructor = Mock(return_value=sampler)
    raw_job = Mock()
    raw_job.job_id.return_value = "quops-ibm-test"
    sampler.run.return_value = raw_job
    monkeypatch.setattr("qiskit_ibm_runtime.SamplerV2", constructor)
    circuits = [_mirror(_workload())]
    jobs = submit_quops_circuits(device, circuits, shots=17, dynamical_decoupling=True)
    assert len(jobs) == 1 and isinstance(jobs[0], QiskitJob)
    options = constructor.call_args.kwargs["options"]
    assert options.dynamical_decoupling.enable is True
    assert options.dynamical_decoupling.sequence_type == "XX"
    assert options.twirling.enable_gates is False
    assert options.twirling.enable_measure is False
    sampler.run.assert_called_once_with(circuits, shots=17)


def test_quantinuum_workload_avoids_redundant_native_roundtrip():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    compiled = compile_quops_circuit(_quantinuum(), circuit, seed=8)
    # One CX already represents one native ZZMax up to local rotations. Lowering
    # it and converting back before mirroring unnecessarily adds local gates.
    assert compiled.circuit.count_ops() == {"cx": 1}


def test_quantinuum_submission_keeps_native_mirror_and_barriers(monkeypatch):
    from pytket.extensions.qiskit import tk_to_qiskit

    device = _quantinuum()
    compiled = compile_quops_circuit(device, _workload(), seed=8)
    mirrors = build_mirror_circuits(compiled.circuit, np.random.default_rng(3))
    assert len({mirror.ideal_bitstring for mirror in mirrors} - {"000"}) == 2
    prepared = [
        prepare_quops_circuit(device, mirror.circuit, compiled.physical_qubits)
        for mirror in mirrors
    ]
    submit = Mock(return_value=SimpleNamespace(id="quantinuum-test"))
    monkeypatch.setattr(device, "submit", submit)
    jobs = submit_quops_circuits(device, prepared, shots=17)
    native_circuits = submit.call_args.args[0]
    for native, mirror, circuit in zip(native_circuits, mirrors, prepared, strict=True):
        # The known workload has two CX gates; each mirror has two halves.
        assert native.n_gates_of_type(OpType.ZZMax) == 4
        assert native.n_gates_of_type(OpType.Barrier) == circuit.count_ops()["barrier"]
        assert {command.op.type for command in native.get_commands()} <= {
            OpType.ZZMax,
            OpType.Rz,
            OpType.PhasedX,
            OpType.Barrier,
            OpType.Measure,
        }
        converted = tk_to_qiskit(native)
        measurements = [
            (converted.find_bit(inst.qubits[0]).index, converted.find_bit(inst.clbits[0]).index)
            for inst in converted.data
            if inst.operation.name == "measure"
        ]
        assert sorted(measurements) == [(0, 0), (1, 1), (2, 2)]
        probabilities = Statevector.from_instruction(
            converted.remove_final_measurements(inplace=False)
        ).probabilities()
        # Independent statevector evaluation checks the native circuit's output
        # and the batch order, with tolerance for floating-point synthesis only.
        assert probabilities[int(mirror.ideal_bitstring, 2)] == pytest.approx(1, rel=0, abs=1e-10)
    assert submit.call_args.kwargs == {"shots": 17, "optimisation_level": 0}
    assert jobs[0].id == "quantinuum-test"


def _braket(*, simulator=False, verbatim=True, native=None):
    device = Mock(spec=BraketDevice)
    device.num_qubits = 5
    device.simulator = simulator
    device._device = SimpleNamespace(
        properties=SimpleNamespace(
            paradigm=SimpleNamespace(nativeGateSet=native or ["rx", "rz", "cz"]),
            action={
                "braket.ir.openqasm.program": SimpleNamespace(
                    supportedPragmas=["braket_verbatim"] if verbatim else [],
                )
            },
        )
    )
    return device


def test_braket_verbatim_preserves_native_gate_count_and_measurement_order():
    # The complete mirror is inside one verbatim box; measurements stay outside.
    circuit = QuantumCircuit(5, 2)
    circuit.rx(0.2, 1)
    circuit.cz(1, 4)
    circuit.barrier(1, 4)
    circuit.cz(1, 4)
    circuit.rx(-0.2, 1)
    circuit.measure([1, 4], [0, 1])
    converted = _braket_circuit(circuit, verbatim=True)
    names = [i.operator.name for i in converted.instructions]
    assert names.count("CZ") == 2
    assert names[0] == "StartVerbatimBox"
    assert names[-3] == "EndVerbatimBox"
    assert names[-2:] == ["Measure", "Measure"]
    device = _braket()
    device.submit.return_value = [SimpleNamespace(id="braket-test")]
    jobs = submit_quops_circuits(device, [circuit], shots=17)
    assert jobs[0].id == "braket-test"
    assert device.submit.call_args.kwargs == {"shots": 17, "disable_qubit_rewiring": True}
    device.transform.assert_not_called()


def test_braket_qpu_preparation_uses_only_fixed_native_rx_angles():
    circuit = _workload()
    prepared = prepare_quops_circuit(_braket(), circuit, [0, 1, 2])
    assert Operator(prepared).equiv(Operator(circuit))
    assert set(prepared.count_ops()) <= {"rx", "rz", "cz"}
    angles = [inst.operation.params[0] for inst in prepared.data if inst.operation.name == "rx"]
    assert angles and set(angles) <= {np.pi / 2, np.pi}


def test_braket_qpu_compiles_routing_before_mirrors():
    device = _braket()
    device._device.topology_graph = nx.path_graph(5)
    compiled = compile_quops_circuit(device, _workload(), physical_qubits=[1, 2, 3], seed=8)
    prepared = prepare_quops_circuit(device, _mirror(compiled.circuit), compiled.physical_qubits)
    assert prepared.count_ops()["cz"] == 2 * compiled.circuit.count_ops()["cz"]
    assert compiled.metadata["native_gates"] == ["cz", "rx", "rz"]
    _assert_zero_state(prepared)
    converted = _braket_circuit(prepared, verbatim=True)
    assert (
        sum(inst.operator.name == "CZ" for inst in converted.instructions)
        == prepared.count_ops()["cz"]
    )


@pytest.mark.parametrize("physical_qubits", [None, [0, 4]], ids=["automatic", "fixed"])
def test_braket_physical_layout_preserves_topology_node_labels(physical_qubits):
    # AWS topology preserves metadata insertion order, which need not be numeric.
    # RustWorkX converts these labels to node payloads, not matching node indices.
    topology = nx.Graph([(3, 1), (1, 4), (4, 0), (0, 2)])
    device = _braket()
    device._device.topology_graph = topology
    workload = QuantumCircuit(2)
    workload.rx(0.2, 0)
    workload.ry(0.4, 1)
    workload.cx(0, 1)

    compiled = compile_quops_circuit(device, workload, physical_qubits=physical_qubits, seed=8)
    assert topology.has_edge(*compiled.physical_qubits)
    if physical_qubits is not None:
        assert compiled.physical_qubits == physical_qubits
    prepared = prepare_quops_circuit(device, _mirror(compiled.circuit), compiled.physical_qubits)
    for instruction in prepared.data:
        if instruction.operation.name == "cz":
            qubits = [prepared.find_bit(q).index for q in instruction.qubits]
            assert topology.has_edge(*qubits)
    _assert_zero_state(prepared)


def test_braket_fully_connected_metadata_without_topology_preserves_physical_ids():
    device = _braket()
    device.id = "mock-native-all-to-all"
    device._provider_name = "rigetti"
    device._device.topology_graph = None
    device._device.properties.paradigm.connectivity = SimpleNamespace(fullyConnected=True)

    compiled = compile_quops_circuit(device, _workload(), physical_qubits=[1, 3, 4], seed=8)

    assert compiled.physical_qubits == [1, 3, 4]
    prepared = prepare_quops_circuit(device, _mirror(compiled.circuit), compiled.physical_qubits)
    _assert_zero_state(prepared)


@pytest.mark.parametrize(
    "device,match",
    [
        (_braket(verbatim=False), "verbatim"),
        (_braket(native=["gpi", "gpi2", "ms"]), "native gate set is unsupported"),
    ],
)
def test_unsupported_braket_qpu_rejected_before_submission(device, match):
    with pytest.raises(ValueError, match=match):
        submit_quops_circuits(device, [_mirror(_workload())], shots=17)
    device.submit.assert_not_called()


def test_braket_simulator_conversion_is_executable_locally():
    from braket.devices import LocalSimulator

    device = _braket(simulator=True)
    compiled = compile_quops_circuit(device, _workload(), seed=3)
    prepared = prepare_quops_circuit(device, _mirror(compiled.circuit), compiled.physical_qubits)
    native = _braket_circuit(prepared, verbatim=False)
    result = LocalSimulator().run(native, shots=32).result()
    assert result.measurement_counts == {"000": 32}


def test_braket_nonzero_output_matches_qiskit_counts_order():
    from braket.devices import LocalSimulator
    from qbraid.runtime.aws.result_builder import BraketGateModelResultBuilder

    circuit = QuantumCircuit(5, 3)
    circuit.rx(np.pi, 1)
    circuit.measure([1, 3, 4], [0, 1, 2])
    native = _braket_circuit(circuit, verbatim=False)
    result = LocalSimulator().run(native, shots=32).result()
    assert result.measurement_counts == {"100": 32}
    assert BraketGateModelResultBuilder(result).get_counts() == {"001": 32}


@pytest.mark.parametrize("basis", ["ideal", "cx", "cz"])
def test_local_submission_reuses_device_and_preserves_prepared_mirrors(monkeypatch, basis):
    backend = AerSimulator()
    if basis != "ideal":
        order = [0, 2, 4, 6, 1, 3, 5]
        edges = [[a, b] for pair in zip(order, order[1:]) for a, b in (pair, pair[::-1])]
        backend = AerSimulator.from_backend(
            GenericBackendV2(7, coupling_map=edges, basis_gates=["rz", "sx", "x", basis], seed=42)
        )
        # Retain the native basis and connectivity, with deterministic ideal outputs.
        backend.set_options(noise_model=None)
    backend.set_options(seed_simulator=42)
    device = LocalAerDevice(provider=Mock(), backend=backend)
    monkeypatch.setattr("metriq_gym.local.job.write", Mock())
    submit = Mock(wraps=device.submit)
    run = Mock(wraps=backend.run)
    monkeypatch.setattr(device, "submit", submit)
    monkeypatch.setattr(backend, "run", run)
    physical = [1, 3, 5]
    compiled = compile_quops_circuit(device, _workload(), seed=8, physical_qubits=physical)
    rng = np.random.default_rng(3)
    mirrors = [*build_mirror_circuits(compiled.circuit, rng), build_spam_circuit(3, rng)]
    assert any(mirror.ideal_bitstring != "000" for mirror in mirrors)
    prepared = [prepare_quops_circuit(device, mirror.circuit, physical) for mirror in mirrors]
    snapshots = [circuit.copy() for circuit in prepared]

    jobs = submit_quops_circuits(device, prepared, shots=32)

    submit.assert_called_once_with(prepared, shots=32)
    run.assert_called_once()
    for before, sent in zip(snapshots, run.call_args.args[0], strict=True):
        # LocalAerDevice's level-0 translation must leave these ISA circuits intact.
        assert sent.data == before.data
        assert sent.global_phase == before.global_phase
        assert sent.num_qubits == before.num_qubits
        assert sent.num_clbits == before.num_clbits
        assert sent.count_ops()["barrier"] > 0
        measurements = [
            (sent.find_bit(inst.qubits[0]).index, sent.find_bit(inst.clbits[0]).index)
            for inst in sent.data
            if inst.operation.name == "measure"
        ]
        assert measurements == [(1, 0), (3, 1), (5, 2)]
    assert len(jobs) == 1
    counts = jobs[0].result().data.measurement_counts
    assert counts == [{mirror.ideal_bitstring: 32} for mirror in mirrors]
    assert qpu_time(jobs[0]) is None


def test_invalid_options_and_layout_fail_before_submission():
    with pytest.raises(ValueError, match="only for IBM"):
        validate_quops_options(_quantinuum(), dynamical_decoupling=True)
    with pytest.raises(ValueError, match="distinct"):
        compile_quops_circuit(_ibm(), _workload(), physical_qubits=[1, 1, 2])
    with pytest.raises(ValueError, match="measurement-free"):
        compile_quops_circuit(_ibm(), _mirror(_workload()))
