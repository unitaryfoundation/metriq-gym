from functools import singledispatch
from typing import cast

import networkx as nx
from qbraid import QuantumDevice
from qbraid.runtime import AzureQuantumDevice, BraketDevice, QiskitBackend


import rustworkx as rx


### Version of a device backend (e.g. ibm_sherbrooke --> '1.6.73') ###
@singledispatch
def version(device: QuantumDevice) -> str:
    raise NotImplementedError(f"Device version not implemented for device of type {type(device)}")


@version.register
def _(device: QiskitBackend) -> str:
    return device._backend.backend_version


@singledispatch
def connectivity_graph(device: QuantumDevice) -> rx.PyGraph:
    raise NotImplementedError(
        f"Connectivity graph not implemented for device of type {type(device)}"
    )


@connectivity_graph.register
def _(device: QiskitBackend) -> rx.PyGraph:
    return device._backend.coupling_map.graph.to_undirected(multigraph=False)


@connectivity_graph.register
def _(device: BraketDevice) -> rx.PyGraph:
    return cast(
        rx.PyGraph,
        rx.networkx_converter(nx.Graph(device._device.topology_graph.to_undirected())),
    )


@connectivity_graph.register
def _(device: AzureQuantumDevice) -> rx.PyGraph:
    return rx.generators.complete_graph(device.metadata()["num_qubits"])
