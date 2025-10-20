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
from metriq_gym.origin.device import OriginDevice
from metriq_gym.quantinuum.device import QuantinuumDevice

if TYPE_CHECKING:  # pragma: no cover
    from pyqpanda3.qcloud import QCloudBackend


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

    edges: list[tuple[int, int]] | None = None
    try:
        backend: QCloudBackend = device.backend
        chip_info = backend.chip_info()
        raw_edges = chip_info.get_chip_topology() if chip_info else None
        if raw_edges:
            unique_edges: set[tuple[int, int]] = set()
            for edge in raw_edges:
                if not edge or len(edge) < 2:
                    continue
                a, b = int(edge[0]), int(edge[1])
                if a == b:
                    continue
                ordered = (min(a, b), max(a, b))
                unique_edges.add(ordered)
            if unique_edges:
                edges = sorted(unique_edges)
    except Exception:
        edges = None

    if not edges:
        return rx.generators.complete_graph(num_qubits)

    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(num_qubits))
    graph.add_edges_from([(a, b, None) for a, b in edges])
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
