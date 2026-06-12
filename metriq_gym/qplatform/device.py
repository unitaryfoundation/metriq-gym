import logging
from functools import singledispatch
from collections.abc import Mapping, Sequence
from datetime import datetime
from numbers import Real
from typing import Any, cast

import networkx as nx
import rustworkx as rx
from qbraid import QuantumDevice
from qbraid.runtime import AzureQuantumDevice, BraketDevice, QiskitBackend
from qiskit.transpiler import CouplingMap
from pytket.architecture import FullyConnected

from metriq_gym.local.device import LocalAerDevice
from metriq_gym.origin.device import OriginDevice, get_origin_connectivity
from metriq_gym.quantinuum.device import QuantinuumDevice


logger = logging.getLogger(__name__)
_MISSING = object()
_MAX_METADATA_WALK_DEPTH = 8


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


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


def _average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value:
        return value
    return None


def _metadata(device: Any) -> Mapping[str, Any]:
    metadata = getattr(device, "metadata", None)
    if not callable(metadata):
        return {}
    try:
        value = metadata()
    except Exception:
        return {}
    return value if isinstance(value, Mapping) else {}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _iter_collection(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _entry_values(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> list[tuple[float, str | None]]:
    if _depth > _MAX_METADATA_WALK_DEPTH:
        return []
    seen = set() if _seen is None else _seen
    if isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        value_id = id(value)
        if value_id in seen:
            return []
        seen.add(value_id)

    if isinstance(value, Mapping):
        unit = value.get("unit")
        unit_str = unit if isinstance(unit, str) else None
        for key in ("value", "mean", "average", "avg"):
            number = _safe_float(value.get(key))
            if number is not None:
                return [(number, unit_str)]
        values: list[tuple[float, str | None]] = []
        for nested_value in value.values():
            values.extend(_entry_values(nested_value, _depth=_depth + 1, _seen=seen))
        return values

    number = _safe_float(value)
    if number is not None:
        return [(number, None)]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for item in value:
            values.extend(_entry_values(item, _depth=_depth + 1, _seen=seen))
        return values

    attr_value = getattr(value, "value", None)
    number = _safe_float(attr_value)
    if number is not None:
        unit = getattr(value, "unit", None)
        return [(number, unit if isinstance(unit, str) else None)]

    return []


def _walk_named_entries(
    value: Any,
    names: set[str],
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> list[tuple[float, str | None]]:
    if _depth > _MAX_METADATA_WALK_DEPTH:
        return []

    entries: list[tuple[float, str | None]] = []
    if isinstance(value, Mapping):
        seen = set() if _seen is None else _seen
        value_id = id(value)
        if value_id in seen:
            return []
        seen.add(value_id)
        for key, nested_value in value.items():
            key_name = str(key).lower()
            if key_name in names:
                entries.extend(_entry_values(nested_value, _seen=set(seen)))
            entries.extend(_walk_named_entries(nested_value, names, _depth=_depth + 1, _seen=seen))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seen = set() if _seen is None else _seen
        value_id = id(value)
        if value_id in seen:
            return []
        seen.add(value_id)
        for item in value:
            entries.extend(_walk_named_entries(item, names, _depth=_depth + 1, _seen=seen))
    return entries


def _seconds(value: float, unit: str | None) -> float:
    normalized_unit = (unit or "s").strip().lower().replace("µ", "u")
    if normalized_unit in {"s", "sec", "second", "seconds"}:
        return value
    if normalized_unit in {"ms", "millisecond", "milliseconds"}:
        return value / 1_000
    if normalized_unit in {"us", "usec", "microsecond", "microseconds"}:
        return value / 1_000_000
    if normalized_unit in {"ns", "nanosecond", "nanoseconds"}:
        return value / 1_000_000_000
    return value


def _error_from_fidelity(fidelity: float) -> float | None:
    if 0 <= fidelity <= 1:
        return 1.0 - fidelity
    if 1 < fidelity <= 100:
        return (100.0 - fidelity) / 100.0
    return None


def _first_average(
    sources: Sequence[Any],
    name_groups: Sequence[set[str]],
    *,
    transform=lambda value, unit: value,
) -> float | None:
    for source in sources:
        for names in name_groups:
            entries = _walk_named_entries(source, names)
            values = []
            for value, unit in entries:
                number = _safe_float(transform(value, unit))
                if number is not None:
                    values.append(number)
            average = _average(values)
            if average is not None:
                return average
    return None


def _first_named_text(
    sources: Sequence[Any],
    names: set[str],
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> str | None:
    if _depth > _MAX_METADATA_WALK_DEPTH:
        return None

    for source in sources:
        if isinstance(source, Mapping):
            seen = set() if _seen is None else _seen
            source_id = id(source)
            if source_id in seen:
                continue
            seen.add(source_id)
            for key, value in source.items():
                if str(key).lower() in names:
                    text = _isoformat(value)
                    if text:
                        return text
                text = _first_named_text([value], names, _depth=_depth + 1, _seen=seen)
                if text:
                    return text
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            seen = set() if _seen is None else _seen
            source_id = id(source)
            if source_id in seen:
                continue
            seen.add(source_id)
            text = _first_named_text(source, names, _depth=_depth + 1, _seen=seen)
            if text:
                return text
    return None


def _qiskit_property_value(entry: Any, name: str) -> tuple[float, str | None] | None:
    if getattr(entry, "name", None) != name:
        return None
    number = _safe_float(getattr(entry, "value", None))
    if number is None:
        return None
    unit = getattr(entry, "unit", None)
    return number, unit if isinstance(unit, str) else None


def _qiskit_qubit_average(
    properties: Any, names: Sequence[str], *, seconds: bool = False
) -> float | None:
    values: list[float] = []
    for qubit_properties in getattr(properties, "qubits", []) or []:
        for entry in qubit_properties:
            for name in names:
                value = _qiskit_property_value(entry, name)
                if value is not None:
                    number, unit = value
                    values.append(_seconds(number, unit) if seconds else number)
                    break
    return _average(values)


def _qiskit_gate_error_average(properties: Any, arity: int) -> float | None:
    values: list[float] = []
    for gate in getattr(properties, "gates", []) or []:
        if len(getattr(gate, "qubits", []) or []) != arity:
            continue
        for entry in getattr(gate, "parameters", []) or []:
            value = _qiskit_property_value(entry, "gate_error")
            if value is not None:
                values.append(value[0])
                break
    return _average(values)


def _fidelity_type_name(fidelity: Any) -> str | None:
    fidelity_type = _field(fidelity, "fidelityType")
    name = _field(fidelity_type, "name")
    return str(name).upper() if name is not None else None


def _fidelity_errors(
    fidelities: Any, *, include_type: str | None = None, exclude_type: str | None = None
) -> list[float]:
    errors: list[float] = []
    entries = _iter_collection(fidelities)
    if not entries and fidelities is not None:
        entries = [fidelities]
    for fidelity_entry in entries:
        fidelity_type = _fidelity_type_name(fidelity_entry)
        if include_type is not None and fidelity_type != include_type:
            continue
        if exclude_type is not None and fidelity_type == exclude_type:
            continue

        raw_fidelity = _field(fidelity_entry, "fidelity", fidelity_entry)
        fidelity = _safe_float(raw_fidelity)
        if fidelity is None:
            continue
        error = _error_from_fidelity(fidelity)
        if error is not None:
            errors.append(error)
    return errors


def _collect_braket_named_average(
    source: Any, names: set[str], *, seconds: bool = False
) -> list[float]:
    entries: list[tuple[float, str | None]] = []
    if not isinstance(source, Mapping) and not (
        isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray))
    ):
        for name in names:
            for candidate in {name, name.lower(), name.upper()}:
                value = _field(source, candidate, _MISSING)
                if value is not _MISSING:
                    entries.extend(_entry_values(value))

    if not entries:
        entries = _walk_named_entries(source, names)

    values = []
    for value, unit in entries:
        values.append(_seconds(value, unit) if seconds else value)
    return values


def _braket_standardized_calibration(standardized: Any) -> dict[str, Any]:
    calibration: dict[str, Any] = {}

    one_qubit_properties = _iter_collection(_field(standardized, "oneQubitProperties"))
    two_qubit_properties = _iter_collection(_field(standardized, "twoQubitProperties"))

    t1_values: list[float] = []
    t2_values: list[float] = []
    readout_errors: list[float] = []
    one_qubit_errors: list[float] = []
    two_qubit_errors: list[float] = []

    for qubit_properties in one_qubit_properties:
        t1_values.extend(_collect_braket_named_average(qubit_properties, {"t1"}, seconds=True))
        t2_values.extend(_collect_braket_named_average(qubit_properties, {"t2"}, seconds=True))
        one_qubit_fidelities = _field(qubit_properties, "oneQubitFidelity")
        readout_errors.extend(_fidelity_errors(one_qubit_fidelities, include_type="READOUT"))
        one_qubit_errors.extend(_fidelity_errors(one_qubit_fidelities, exclude_type="READOUT"))
        readout_errors.extend(_fidelity_errors(_field(qubit_properties, "readoutFidelity")))
        one_qubit_errors.extend(_fidelity_errors(_field(qubit_properties, "singleQubitFidelity")))

    if not t1_values:
        t1_values.extend(_collect_braket_named_average(standardized, {"t1"}, seconds=True))
    if not t2_values:
        t2_values.extend(_collect_braket_named_average(standardized, {"t2"}, seconds=True))
    if not readout_errors:
        readout_errors.extend(_fidelity_errors(_field(standardized, "readoutFidelity")))
    if not one_qubit_errors:
        one_qubit_errors.extend(_fidelity_errors(_field(standardized, "singleQubitFidelity")))

    for edge_properties in two_qubit_properties:
        two_qubit_errors.extend(_fidelity_errors(_field(edge_properties, "twoQubitGateFidelity")))
    if not two_qubit_errors:
        two_qubit_errors.extend(_fidelity_errors(_field(standardized, "twoQubitGateFidelity")))

    t1 = _average(t1_values)
    if t1 is not None:
        calibration["avg_t1_s"] = t1

    t2 = _average(t2_values)
    if t2 is not None:
        calibration["avg_t2_s"] = t2

    readout = _average(readout_errors)
    if readout is not None:
        calibration["avg_readout_error"] = readout

    one_qubit_error = _average(one_qubit_errors)
    if one_qubit_error is not None:
        calibration["avg_1q_gate_error"] = one_qubit_error

    two_qubit_error = _average(two_qubit_errors)
    if two_qubit_error is not None:
        calibration["avg_2q_gate_error"] = two_qubit_error

    last_update_date = _first_named_text(
        [standardized], {"updatedat", "lastupdatedate", "last_update_date", "lastupdated"}
    )
    if not last_update_date:
        # Braket SDK model objects expose attributes instead of dict keys.
        for name in ("updatedAt", "lastUpdateDate", "last_update_date", "lastUpdated"):
            last_update_date = _isoformat(_field(standardized, name))
            if last_update_date:
                break
    if last_update_date:
        calibration["last_update_date"] = last_update_date

    return calibration


@singledispatch
def calibration_metadata(device: QuantumDevice) -> dict[str, Any]:
    """Return normalized vendor calibration metadata when available."""
    return {}


@calibration_metadata.register
def _(device: QiskitBackend) -> dict[str, Any]:
    backend = device._backend
    properties_func = getattr(backend, "properties", None)
    properties = properties_func() if callable(properties_func) else None
    if properties is None:
        return {}

    calibration: dict[str, Any] = {}

    t1 = _qiskit_qubit_average(properties, ("T1", "t1"), seconds=True)
    if t1 is not None:
        calibration["avg_t1_s"] = t1

    t2 = _qiskit_qubit_average(properties, ("T2", "t2"), seconds=True)
    if t2 is not None:
        calibration["avg_t2_s"] = t2

    readout = _qiskit_qubit_average(properties, ("readout_error",))
    if readout is None:
        readout = _qiskit_qubit_average(properties, ("prob_meas1_prep0", "prob_meas0_prep1"))
    if readout is not None:
        calibration["avg_readout_error"] = readout

    one_qubit_error = _qiskit_gate_error_average(properties, 1)
    if one_qubit_error is not None:
        calibration["avg_1q_gate_error"] = one_qubit_error

    two_qubit_error = _qiskit_gate_error_average(properties, 2)
    if two_qubit_error is not None:
        calibration["avg_2q_gate_error"] = two_qubit_error

    last_update_date = _isoformat(getattr(properties, "last_update_date", None))
    if last_update_date:
        calibration["last_update_date"] = last_update_date

    return calibration


@calibration_metadata.register
def _(device: BraketDevice) -> dict[str, Any]:
    device_properties = getattr(getattr(device, "_device", None), "properties", None)
    device_metadata = _metadata(device)
    metadata_calibration = device_metadata.get("calibration")
    provider_name = str(getattr(device, "_provider_name", "") or "").lower()
    is_rigetti = "rigetti" in provider_name
    is_ionq = "ionq" in provider_name

    for source in (device_properties, device_metadata, metadata_calibration):
        standardized = _field(source, "standardized")
        standardized_calibration = _braket_standardized_calibration(standardized)
        if standardized_calibration:
            return standardized_calibration

    sources = [source for source in (device_properties, device_metadata) if source]

    calibration: dict[str, Any] = {}
    t1 = _first_average(
        sources,
        [{"t1", "avg_t1_s"}],
        transform=lambda value, unit: _seconds(value, unit),
    )
    if t1 is not None:
        calibration["avg_t1_s"] = t1

    t2 = _first_average(
        sources,
        [{"t2", "avg_t2_s"}],
        transform=lambda value, unit: _seconds(value, unit),
    )
    if t2 is not None:
        calibration["avg_t2_s"] = t2

    readout = _first_average(
        sources,
        [{"readouterror", "readout_error", "avg_readout_error"}],
    )
    if readout is None:
        readout_fidelity_names = [{"readoutfidelity", "fidelityreadout"}]
        if is_rigetti:
            readout_fidelity_names.append({"fro"})
        if is_ionq:
            readout_fidelity_names.append({"spam"})
        readout = _first_average(
            sources,
            readout_fidelity_names,
            transform=lambda value, unit: _error_from_fidelity(value),
        )
    if readout is not None:
        calibration["avg_readout_error"] = readout

    one_qubit_error = _first_average(
        sources,
        [{"onequbitgateerror", "singlequbitgateerror", "avg_1q_gate_error"}],
    )
    if one_qubit_error is None:
        one_qubit_fidelity_names = [
            {"onequbitfidelity"},
            {"singlequbitgatefidelity"},
            {"singlequbitfidelity"},
        ]
        if is_rigetti:
            one_qubit_fidelity_names.append({"f1qrb"})
        one_qubit_error = _first_average(
            sources,
            one_qubit_fidelity_names,
            transform=lambda value, unit: _error_from_fidelity(value),
        )
    if one_qubit_error is not None:
        calibration["avg_1q_gate_error"] = one_qubit_error

    two_qubit_error = _first_average(
        sources,
        [{"twoqubitgateerror", "avg_2q_gate_error"}],
    )
    if two_qubit_error is None:
        two_qubit_fidelity_names = [{"twoqubitfidelity"}, {"twoqubitgatefidelity"}]
        if is_rigetti:
            two_qubit_fidelity_names.append({"fcz"})
        two_qubit_error = _first_average(
            sources,
            two_qubit_fidelity_names,
            transform=lambda value, unit: _error_from_fidelity(value),
        )
    if two_qubit_error is not None:
        calibration["avg_2q_gate_error"] = two_qubit_error

    last_update_date = _first_named_text(
        sources, {"lastupdatedate", "last_update_date", "lastupdated", "timestamp"}
    )
    if last_update_date:
        calibration["last_update_date"] = last_update_date

    return calibration


@calibration_metadata.register
def _(device: AzureQuantumDevice) -> dict[str, Any]:
    # TODO(#668): normalize Azure calibration metadata when qBraid exposes it.
    return {}


@calibration_metadata.register
def _(device: LocalAerDevice) -> dict[str, Any]:
    # TODO(#668): local simulators do not expose vendor calibration metadata.
    return {}


@calibration_metadata.register
def _(device: OriginDevice) -> dict[str, Any]:
    # TODO(#668): normalize Origin calibration metadata when available.
    return {}


@calibration_metadata.register
def _(device: QuantinuumDevice) -> dict[str, Any]:
    # TODO(#668): normalize Quantinuum calibration metadata when available.
    return {}


def normalized_metadata(device: QuantumDevice) -> dict:
    """Return a minimal, normalized subset of device metadata.

    Includes only the following keys when available:
    - simulator: bool
    - version: str
    - num_qubits: int
    - calibration: dict
    """
    meta: dict = {}
    try:
        simulator = getattr(getattr(device, "profile", object()), "simulator", None)
        if isinstance(simulator, bool):
            meta["simulator"] = simulator
    except Exception:
        logger.debug(
            "Failed to extract simulator metadata for %s", type(device).__name__, exc_info=True
        )

    try:
        n = getattr(device, "num_qubits", None)
        if isinstance(n, int):
            meta["num_qubits"] = n
    except Exception:
        logger.debug(
            "Failed to extract qubit count metadata for %s", type(device).__name__, exc_info=True
        )

    try:
        ver = version(device)
        if isinstance(ver, str) and ver:
            meta["version"] = ver
    except Exception:
        logger.debug(
            "Failed to extract version metadata for %s", type(device).__name__, exc_info=True
        )

    try:
        calibration = calibration_metadata(device)
        if calibration:
            meta["calibration"] = calibration
    except Exception:
        logger.debug(
            "Failed to extract calibration metadata for %s", type(device).__name__, exc_info=True
        )

    return meta
