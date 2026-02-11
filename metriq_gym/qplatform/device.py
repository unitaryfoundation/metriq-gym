from functools import singledispatch
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

    return meta
