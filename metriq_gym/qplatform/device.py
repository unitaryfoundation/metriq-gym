from functools import singledispatch
from typing import cast

import networkx as nx
from qbraid import QuantumDevice
from qbraid.runtime import AzureQuantumDevice, BraketDevice, QiskitBackend
from qiskit.transpiler import CouplingMap


import rustworkx as rx

from metriq_gym.local.device import LocalAerDevice


### Version of a device backend (e.g. ibm_sherbrooke --> '1.6.73') ###
@singledispatch
def version(device: QuantumDevice) -> str:
    raise NotImplementedError(f"Device version not implemented for device of type {type(device)}")


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
    def create_all_to_all_digraph(n_nodes):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    graph.add_edge(i, j)
        return graph

    if device._provider_name in ["Amazon Braket", "IonQ"]:
        device_topology = create_all_to_all_digraph(device.num_qubits)
    else:
        device_topology = device._device.topology_graph

    return cast(
        rx.PyGraph,
        rx.networkx_converter(nx.Graph(device_topology.to_undirected())),
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
