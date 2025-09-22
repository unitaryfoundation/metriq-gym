from functools import singledispatch
from typing import cast

import networkx as nx
import rustworkx as rx
import qnexus as qnx
from qbraid import QuantumDevice
from qbraid.runtime import AzureQuantumDevice, BraketDevice, QiskitBackend
from qiskit.transpiler import CouplingMap

from metriq_gym.local.device import LocalAerDevice
from metriq_gym.quantinuum.device import QuantinuumDevice


# Version of a device backend (e.g. ibm_sherbrooke --> '1.6.73').
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
    return cast(
        rx.PyGraph,
        rx.networkx_converter(nx.Graph(device._device.topology_graph.to_undirected())),
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
    df = qnx.devices.get_all(issuers=[qnx.devices.IssuerEnum.QUANTINUUM]).df()
    df["n_qubits"] = [
        len(bi.architecture.nodes) if hasattr(bi.architecture, "nodes")
        else (_ for _ in ()).throw(RuntimeError("Could not determine qubit count"))
        for bi in df["backend_info"]
    ]
    device_name = device.profile.device_id
    row = df[df["device_name"] == device_name].iloc[0]
    num_qubits = row["n_qubits"]

    return rx.generators.complete_graph(num_qubits)
