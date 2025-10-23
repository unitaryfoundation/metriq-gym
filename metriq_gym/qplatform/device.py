from functools import singledispatch
from typing import cast, TYPE_CHECKING

import networkx as nx
import rustworkx as rx
import qnexus as qnx
from qbraid import QuantumDevice
from qbraid.runtime import AzureQuantumDevice, BraketDevice, QiskitBackend
from qiskit.transpiler import CouplingMap
from pytket.architecture import FullyConnected

from metriq_gym.local.device import LocalAerDevice
from metriq_gym.origin.device import OriginDevice, get_origin_connectivity
from metriq_gym.quantinuum.device import QuantinuumDevice

if TYPE_CHECKING:  # pragma: no cover
    pass


# Version of a device backend (e.g. ibm_sherbrooke --> '1.6.73').
@singledispatch
def version(device: QuantumDevice) -> str:
    raise NotImplementedError(f"Device version not implemented for device of type {type(device)}")


@version.register
def _(device: QuantinuumDevice) -> str:
    device_name = device.profile.device_id

    df = qnx.devices.get_all(issuers=[qnx.devices.IssuerEnum.QUANTINUUM]).df()
    row = df.loc[df["device_name"] == device_name].iloc[0]

    backend_info = row["backend_info"]
    return backend_info.version


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
    device_name = device.profile.device_id

    df = qnx.devices.get_all(issuers=[qnx.devices.IssuerEnum.QUANTINUUM]).df()
    row = df.loc[df["device_name"] == device_name].iloc[0]

    arch = row["backend_info"].architecture
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

    active_nodes, raw_edges = get_origin_connectivity(device)

    active_set = set(active_nodes)
    filtered_edges = [
        (a, b) for a, b in raw_edges if not active_set or (a in active_set and b in active_set)
    ]

    if active_nodes and not filtered_edges and raw_edges:
        filtered_edges = raw_edges
        active_nodes = sorted({node for edge in filtered_edges for node in edge})

    if active_nodes:
        node_labels = active_nodes
    elif filtered_edges:
        node_labels = sorted({node for edge in filtered_edges for node in edge})
    else:
        node_labels = []

    if not node_labels:
        size = num_qubits if isinstance(num_qubits, int) and num_qubits > 0 else 0
        if size <= 0:
            return rx.PyGraph(multigraph=False)
        return rx.generators.complete_graph(size)

    node_map = {node: idx for idx, node in enumerate(node_labels)}
    mapped_edges = [
        (node_map[a], node_map[b], None)
        for a, b in filtered_edges
        if a in node_map and b in node_map
    ]

    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(len(node_labels)))
    if mapped_edges:
        graph.add_edges_from(mapped_edges)
    return graph


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
