from functools import singledispatch
import math
from typing import cast

import networkx as nx
import rustworkx as rx
from qbraid import QuantumDevice
from qbraid.runtime import AzureQuantumDevice, BraketDevice, QiskitBackend
from qiskit.transpiler import CouplingMap
from pytket.architecture import FullyConnected

from metriq_gym.local.device import LocalAerDevice
from metriq_gym.origin.device import OriginDevice, get_origin_connectivity
from metriq_gym.quantinuum.device import QuantinuumDevice


# Version of a device backend (e.g. ibm_sherbrooke --> '1.6.73').
@singledispatch
def version(device: QuantumDevice) -> str:
    raise NotImplementedError(f"Device version not implemented for device of type {type(device)}")


@version.register
def _(device: QuantinuumDevice) -> str:
    return device._backend_info.version


@version.register
def _(device: QiskitBackend) -> str:
    return device._backend.backend_version


@version.register
def _(device: LocalAerDevice) -> str:
    return device._backend.configuration().backend_version


def _mean(values: list[float]) -> float | None:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _field(container, name: str, default=None):
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def _numeric_value(value) -> float | None:
    if isinstance(value, bool) or isinstance(value, str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _error_from_fidelity(value) -> float | None:
    fidelity = _numeric_value(value)
    if fidelity is None:
        return None
    if 0 <= fidelity <= 1:
        return 1 - fidelity
    if 1 < fidelity <= 100:
        return (100 - fidelity) / 100
    return None


def _parameter_value(parameter, name: str) -> float | None:
    if _field(parameter, "name") != name:
        return None
    return _numeric_value(_field(parameter, "value"))


def _duration_seconds(duration) -> float | None:
    value = _numeric_value(_field(duration, "value"))
    if value is None:
        return None
    unit = _field(duration, "unit", "s")
    unit = str(getattr(unit, "value", unit)).lower()
    multipliers = {
        "s": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    return value * multiplier


def _extend_fidelity_errors(errors: list[float], fidelities) -> None:
    for fidelity in fidelities or []:
        error = _error_from_fidelity(_field(fidelity, "fidelity"))
        if error is not None:
            errors.append(error)


def _fidelity_type_name(fidelity) -> str | None:
    fidelity_type = _field(fidelity, "fidelityType")
    name = _field(fidelity_type, "name")
    if name is None:
        return None
    return str(name).upper()


def _extend_one_qubit_standardized_errors(
    *,
    readout_errors: list[float],
    one_qubit_gate_errors: list[float],
    fidelities,
) -> None:
    for fidelity in fidelities or []:
        error = _error_from_fidelity(_field(fidelity, "fidelity"))
        if error is None:
            continue
        if _fidelity_type_name(fidelity) == "READOUT":
            readout_errors.append(error)
        else:
            one_qubit_gate_errors.append(error)


def _set_if_present(metadata: dict, key: str, value: float | str | None) -> None:
    if value is not None:
        metadata[key] = value


def _summary_metadata(
    *,
    t1_values: list[float] | None = None,
    t2_values: list[float] | None = None,
    readout_errors: list[float] | None = None,
    one_qubit_gate_errors: list[float] | None = None,
    two_qubit_gate_errors: list[float] | None = None,
    last_update=None,
) -> dict:
    metadata: dict = {}
    _set_if_present(metadata, "avg_t1_s", _mean(t1_values or []))
    _set_if_present(metadata, "avg_t2_s", _mean(t2_values or []))
    _set_if_present(metadata, "avg_readout_error", _mean(readout_errors or []))
    _set_if_present(metadata, "avg_1q_gate_error", _mean(one_qubit_gate_errors or []))
    _set_if_present(metadata, "avg_2q_gate_error", _mean(two_qubit_gate_errors or []))
    if hasattr(last_update, "isoformat"):
        metadata["last_update_date"] = last_update.isoformat()
    elif isinstance(last_update, str) and last_update:
        metadata["last_update_date"] = last_update
    return metadata


@singledispatch
def calibration_metadata(device: QuantumDevice) -> dict:
    """Return normalized vendor calibration data when a provider exposes it.

    The values are intentionally aggregated to keep exported benchmark payloads
    compact and comparable across providers.
    """
    return {}


@calibration_metadata.register
def _(device: QiskitBackend) -> dict:
    backend = device._backend
    if not hasattr(backend, "properties"):
        return {}

    props = backend.properties()
    if props is None:
        return {}

    t1_values: list[float] = []
    t2_values: list[float] = []
    readout_errors: list[float] = []
    for qubit_properties in getattr(props, "qubits", []) or []:
        for parameter in qubit_properties:
            t1 = _parameter_value(parameter, "T1")
            if t1 is not None:
                t1_values.append(t1)
            t2 = _parameter_value(parameter, "T2")
            if t2 is not None:
                t2_values.append(t2)
            readout_error = _parameter_value(parameter, "readout_error")
            if readout_error is not None:
                readout_errors.append(readout_error)

    one_qubit_gate_errors: list[float] = []
    two_qubit_gate_errors: list[float] = []
    for gate_properties in getattr(props, "gates", []) or []:
        gate_error = next(
            (
                value
                for parameter in getattr(gate_properties, "parameters", []) or []
                if (value := _parameter_value(parameter, "gate_error")) is not None
            ),
            None,
        )
        if gate_error is None:
            continue
        qubit_count = len(getattr(gate_properties, "qubits", []) or [])
        if qubit_count == 1:
            one_qubit_gate_errors.append(gate_error)
        elif qubit_count == 2:
            two_qubit_gate_errors.append(gate_error)

    return _summary_metadata(
        t1_values=t1_values,
        t2_values=t2_values,
        readout_errors=readout_errors,
        one_qubit_gate_errors=one_qubit_gate_errors,
        two_qubit_gate_errors=two_qubit_gate_errors,
        last_update=getattr(props, "last_update_date", None),
    )


def _braket_standardized_calibration(standardized) -> dict:
    t1_values: list[float] = []
    t2_values: list[float] = []
    readout_errors: list[float] = []
    one_qubit_gate_errors: list[float] = []
    two_qubit_gate_errors: list[float] = []

    t1 = _duration_seconds(_field(standardized, "T1"))
    if t1 is not None:
        t1_values.append(t1)
    t2 = _duration_seconds(_field(standardized, "T2"))
    if t2 is not None:
        t2_values.append(t2)

    _extend_fidelity_errors(readout_errors, _field(standardized, "readoutFidelity"))
    _extend_fidelity_errors(one_qubit_gate_errors, _field(standardized, "singleQubitFidelity"))
    _extend_fidelity_errors(two_qubit_gate_errors, _field(standardized, "twoQubitGateFidelity"))

    one_qubit_properties = _field(standardized, "oneQubitProperties") or {}
    for qubit_properties in one_qubit_properties.values():
        _extend_one_qubit_standardized_errors(
            readout_errors=readout_errors,
            one_qubit_gate_errors=one_qubit_gate_errors,
            fidelities=_field(qubit_properties, "oneQubitFidelity"),
        )

    two_qubit_properties = _field(standardized, "twoQubitProperties") or {}
    for edge_properties in two_qubit_properties.values():
        _extend_fidelity_errors(
            two_qubit_gate_errors,
            _field(edge_properties, "twoQubitGateFidelity"),
        )

    return _summary_metadata(
        t1_values=t1_values,
        t2_values=t2_values,
        readout_errors=readout_errors,
        one_qubit_gate_errors=one_qubit_gate_errors,
        two_qubit_gate_errors=two_qubit_gate_errors,
        last_update=_field(standardized, "updatedAt"),
    )


def _braket_rigetti_calibration(provider) -> dict:
    specs = _field(provider, "specs") or {}
    one_qubit = specs.get("1Q", {})
    two_qubit = specs.get("2Q", {})

    t1_values = []
    t2_values = []
    readout_errors = []
    one_qubit_gate_errors = []
    two_qubit_gate_errors = []

    for qubit_specs in one_qubit.values():
        t1 = _numeric_value(_field(qubit_specs, "T1"))
        if t1 is not None:
            t1_values.append(t1)
        t2 = _numeric_value(_field(qubit_specs, "T2"))
        if t2 is not None:
            t2_values.append(t2)
        readout_error = _error_from_fidelity(_field(qubit_specs, "fRO"))
        if readout_error is not None:
            readout_errors.append(readout_error)
        one_qubit_gate_error = _error_from_fidelity(_field(qubit_specs, "f1QRB"))
        if one_qubit_gate_error is not None:
            one_qubit_gate_errors.append(one_qubit_gate_error)

    for gate_specs in two_qubit.values():
        gate_error = _error_from_fidelity(_field(gate_specs, "fCZ"))
        if gate_error is not None:
            two_qubit_gate_errors.append(gate_error)

    return _summary_metadata(
        t1_values=t1_values,
        t2_values=t2_values,
        readout_errors=readout_errors,
        one_qubit_gate_errors=one_qubit_gate_errors,
        two_qubit_gate_errors=two_qubit_gate_errors,
    )


def _braket_ionq_calibration(provider) -> dict:
    fidelity = _field(provider, "fidelity") or {}
    readout_errors = []
    one_qubit_gate_errors = []
    two_qubit_gate_errors = []

    one_qubit_gate_error = _error_from_fidelity(_field(fidelity.get("1Q", {}), "mean"))
    if one_qubit_gate_error is not None:
        one_qubit_gate_errors.append(one_qubit_gate_error)
    two_qubit_gate_error = _error_from_fidelity(_field(fidelity.get("2Q", {}), "mean"))
    if two_qubit_gate_error is not None:
        two_qubit_gate_errors.append(two_qubit_gate_error)
    readout_error = _error_from_fidelity(_field(fidelity.get("spam", {}), "mean"))
    if readout_error is not None:
        readout_errors.append(readout_error)

    return _summary_metadata(
        readout_errors=readout_errors,
        one_qubit_gate_errors=one_qubit_gate_errors,
        two_qubit_gate_errors=two_qubit_gate_errors,
    )


@calibration_metadata.register
def _(device: BraketDevice) -> dict:
    properties = getattr(device._device, "properties", None)
    if properties is None:
        return {}

    metadata = _braket_standardized_calibration(_field(properties, "standardized"))
    if metadata:
        return metadata

    provider = _field(properties, "provider")
    provider_name = type(provider).__name__.lower()
    if "rigetti" in provider_name:
        return _braket_rigetti_calibration(provider)
    if "ionq" in provider_name:
        return _braket_ionq_calibration(provider)
    return {}


def coupling_map_to_graph(coupling_map: CouplingMap) -> rx.PyGraph:
    return coupling_map.graph.to_undirected(multigraph=False)


@singledispatch
def connectivity_graph(device: QuantumDevice) -> rx.PyGraph:
    raise NotImplementedError(
        f"Connectivity graph not implemented for device of type {type(device)}"
    )


@connectivity_graph.register
def _(device: QiskitBackend) -> rx.PyGraph:
    return coupling_map_to_graph(device._backend.coupling_map)


@connectivity_graph.register
def _(device: BraketDevice) -> rx.PyGraph:
    if device._provider_name == "Amazon Braket":
        device_topology = nx.complete_graph(device.num_qubits)
    else:
        device_topology = device._device.topology_graph.to_undirected()

    return cast(
        rx.PyGraph,
        rx.networkx_converter(nx.Graph(device_topology)),
    )


@connectivity_graph.register
def _(device: AzureQuantumDevice) -> rx.PyGraph:
    return rx.generators.complete_graph(device.metadata()["num_qubits"])


@connectivity_graph.register
def _(device: LocalAerDevice) -> rx.PyGraph:
    coupling_list = device._backend.configuration().coupling_map
    if coupling_list is None:
        return rx.generators.complete_graph(device._backend.configuration().n_qubits)
    return coupling_map_to_graph(CouplingMap(coupling_list))


@connectivity_graph.register
def _(device: QuantinuumDevice) -> rx.PyGraph:
    arch = device._backend_info.architecture
    num_qubits = len(arch.nodes)

    is_fc = isinstance(arch, FullyConnected)
    if not is_fc and hasattr(arch, "edges"):
        is_fc = len(arch.edges) == num_qubits * (num_qubits - 1) // 2

    if is_fc:
        # safe to generate a complete graph
        return rx.generators.complete_graph(num_qubits)
    else:
        # build graph from actual connectivity
        g = rx.PyGraph()
        g.add_nodes_from(range(num_qubits))
        # arch.edges contains Node objects; map them to indices
        node_index = {node: i for i, node in enumerate(arch.nodes)}
        g.add_edges_from([(node_index[a], node_index[b], None) for (a, b) in arch.edges])
        return g


@connectivity_graph.register
def _(device: OriginDevice) -> rx.PyGraph:
    num_qubits = device.num_qubits
    if not isinstance(num_qubits, int):
        raise NotImplementedError(
            "Origin device does not report a qubit count for connectivity graph"
        )

    if getattr(device.profile, "simulator", False):
        return rx.generators.complete_graph(num_qubits)

    available_qubits, edges = get_origin_connectivity(device)
    graph = rx.PyGraph()
    graph.add_nodes_from(available_qubits)
    node_index = {node: i for i, node in enumerate(available_qubits)}
    graph.add_edges_from([(node_index[a], node_index[b], None) for (a, b) in edges])
    return graph


@singledispatch
def connectivity_graph_for_gate(device: QuantumDevice, gate: str) -> rx.PyGraph | None:
    """Return connectivity graph that works for the given gate, if available for the device."""
    return None


@connectivity_graph_for_gate.register
def _(device: QiskitBackend, gate: str) -> rx.PyGraph | None:
    if gate in device._backend.target:
        return coupling_map_to_graph(device._backend.target.build_coupling_map(two_q_gate=gate))
    return None


@singledispatch
def pruned_connectivity_graph(device: QuantumDevice, graph: rx.PyGraph) -> rx.PyGraph:
    """Return a new graph with faulty qubits and edges removed.

    For devices that don't support faulty qubit/edge reporting, returns
    a copy of the input graph unchanged.

    Args:
        device: The quantum device to check for faulty components.
        graph: The connectivity graph to filter.

    Returns:
        A new PyGraph with faulty qubits (nodes) and faulty gate edges removed.
        Node indices are preserved - removed nodes leave gaps in the index space.
    """
    return graph.copy()


@pruned_connectivity_graph.register
def _(device: QiskitBackend, graph: rx.PyGraph) -> rx.PyGraph:
    backend = device._backend

    if not hasattr(backend, "properties") or backend.properties() is None:
        return graph.copy()

    props = backend.properties()

    faulty_gates = backend.properties().faulty_gates()
    faulty_edges = [tuple(gate.qubits) for gate in faulty_gates if len(gate.qubits) > 1]

    new_graph = graph.copy()
    new_graph.remove_edges_from(faulty_edges)
    new_graph.remove_nodes_from(props.faulty_qubits())

    return new_graph


def normalized_metadata(device: QuantumDevice) -> dict:
    """Return a minimal, normalized subset of device metadata.

    Includes only the following keys when available:
    - simulator: bool
    - version: str
    - num_qubits: int
    """
    meta: dict = {}
    try:
        simulator = getattr(getattr(device, "profile", object()), "simulator", None)
        if isinstance(simulator, bool):
            meta["simulator"] = simulator
    except Exception:
        pass

    try:
        n = getattr(device, "num_qubits", None)
        if isinstance(n, int):
            meta["num_qubits"] = n
    except Exception:
        pass

    try:
        ver = version(device)
        if isinstance(ver, str) and ver:
            meta["version"] = ver
    except Exception:
        pass

    try:
        calibration = calibration_metadata(device)
        if calibration:
            meta["calibration"] = calibration
    except Exception:
        pass

    return meta
