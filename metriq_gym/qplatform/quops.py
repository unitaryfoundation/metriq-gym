"""Compilation and submission boundaries for experimental QUOPS.

Optimize the test computation before constructing its mirrors. Once mirrors
exist, only basis translation is permitted. In particular, Braket QPUs need
verbatim execution: losing the Qiskit barriers in a normal conversion would
allow the provider to cancel a circuit against its inverse.
"""

from dataclasses import dataclass, field
from importlib.metadata import version
import math
from typing import Any

from qbraid.runtime import BraketDevice, QiskitBackend, QiskitJob
from qbraid.runtime.quantinuum import QuantinuumDevice
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, Target

from metriq_gym.local.device import LocalAerDevice
from metriq_gym.qplatform.device import connectivity_graph


@dataclass
class CompiledQUOPSCircuit:
    circuit: QuantumCircuit
    physical_qubits: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_quops_options(device, *, dynamical_decoupling: bool = False) -> None:
    """Reject unsupported execution choices during estimation as well as dispatch."""
    if dynamical_decoupling and not isinstance(device, QiskitBackend):
        raise ValueError("QUOPS dynamical_decoupling is available only for IBM")
    if isinstance(device, BraketDevice):
        _braket_basis(device)
    elif isinstance(device, QuantinuumDevice):
        if str(device.id).lower().startswith("helios"):
            raise ValueError(
                "QUOPS Helios execution requires a dedicated Helios adapter; use an H-series device"
            )
    elif not isinstance(device, (QiskitBackend, LocalAerDevice)):
        raise ValueError(f"QUOPS does not support device type {type(device).__name__}")


def _layout(width: int, physical_qubits: list[int] | None, capacity: int) -> list[int]:
    layout = list(range(width)) if physical_qubits is None else list(physical_qubits)
    if len(layout) != width or len(set(layout)) != width:
        raise ValueError("QUOPS physical_qubits must contain one distinct index per data qubit")
    if any(type(q) is not int or q < 0 or q >= capacity for q in layout):
        raise ValueError("QUOPS physical_qubits contains an unavailable qubit")
    # Keep classical result order consistent across Qiskit, pytket and Braket.
    return sorted(layout)


def _restricted_target(target: Target, physical: list[int]) -> Target:
    """Retain actual native gates/calibrations on the selected physical subset."""
    inverse = {q: i for i, q in enumerate(physical)}
    subset = Target(num_qubits=len(physical), dt=target.dt)
    for name in target.operation_names:
        operation = target.operation_from_name(name)
        # Control-flow classes have global arity and are irrelevant to these
        # static circuits; Target forbids attaching per-qubit properties to them.
        if isinstance(operation, type):
            continue
        properties: dict[tuple[int, ...] | None, Any] = {}
        for qargs, props in target[name].items():
            if qargs is None:
                properties[None] = props
            elif all(q in inverse for q in qargs):
                properties[tuple(inverse[q] for q in qargs)] = props
        if properties:
            subset.add_instruction(operation, properties, name=name)
    return subset


def _compact(circuit: QuantumCircuit, physical: list[int]) -> QuantumCircuit:
    inverse = {q: i for i, q in enumerate(physical)}
    compact = QuantumCircuit(len(physical))
    compact.global_phase = circuit.global_phase
    for inst in circuit.data:
        indices = [circuit.find_bit(q).index for q in inst.qubits]
        if any(q not in inverse for q in indices):
            if inst.operation.name == "barrier":
                continue
            raise ValueError("QUOPS compilation used auxiliary routing qubits")
        compact.append(inst.operation, [inverse[q] for q in indices])
    return compact


def _braket_basis(device: BraketDevice) -> list[str]:
    if device.simulator:
        return ["rx", "ry", "rz", "cx"]
    properties = device._device.properties
    actions = getattr(properties, "action", {})
    action = actions.get("braket.ir.openqasm.program")
    pragmas = getattr(action, "supportedPragmas", []) or []
    native = {name.lower() for name in getattr(properties.paradigm, "nativeGateSet", [])}
    if "braket_verbatim" not in pragmas:
        raise ValueError("QUOPS requires Braket QPU verbatim execution to preserve mirror circuits")
    # Rigetti accepts only fixed native Rx angles. Synthesize with SX/X first,
    # then express these pulses as Rx(pi/2)/Rx(pi) before submission.
    if {"rx", "rz", "cz"} <= native:
        return ["sx", "x", "rz", "cz"]
    if {"rx", "rz", "cnot"} <= native:
        return ["sx", "x", "rz", "cx"]
    raise ValueError(
        "QUOPS currently supports Braket QPUs with native rx/rz/cz or rx/rz/cnot "
        "and verbatim execution; this device's native gate set is unsupported"
    )


def _physical_node_ids(device, graph) -> dict[int, int]:
    """Resolve physical IDs; Braket graph indices follow topology insertion order."""
    if isinstance(device, BraketDevice):
        # Generated complete graphs have no payload; their indices already are IDs.
        return {
            node: node if graph[node] is None else int(graph[node]) for node in graph.node_indices()
        }
    return {node: node for node in graph.node_indices()}


def _coupling(device, physical: list[int] | None = None) -> CouplingMap | None:
    if isinstance(device, QuantinuumDevice) or device.simulator:
        return None
    graph = connectivity_graph(device)
    physical_ids = _physical_node_ids(device, graph)
    nodes = sorted(physical_ids.values()) if physical is None else physical
    inverse = {q: i for i, q in enumerate(nodes)}
    edges = [
        (inverse[physical_ids[a]], inverse[physical_ids[b]])
        for a, b in graph.edge_list()
        if physical_ids[a] in inverse and physical_ids[b] in inverse
    ]
    result = CouplingMap()
    for q in range(len(nodes)):
        result.add_physical_qubit(q)
    for a, b in edges:
        result.add_edge(a, b)
        result.add_edge(b, a)
    return result


def _quantinuum_native(circuit: QuantumCircuit):
    from pytket.circuit import OpType
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.passes import AutoRebase

    native = qiskit_to_tk(circuit)
    AutoRebase({OpType.ZZMax, OpType.Rz, OpType.PhasedX}).apply(native)
    return native


def _braket_fixed_rx(circuit: QuantumCircuit) -> QuantumCircuit:
    native = circuit.copy_empty_like()
    for inst in circuit.data:
        if inst.operation.name in {"sx", "x"}:
            angle = math.pi / 2 if inst.operation.name == "sx" else math.pi
            native.rx(angle, inst.qubits[0])
            native.global_phase += angle / 2
        else:
            native.append(inst)
    return native


def _connected_subset(device, width: int, preferred: list[int] | None = None) -> list[int]:
    graph = connectivity_graph(device)
    physical_ids = _physical_node_ids(device, graph)
    node_indices = {physical: node for node, physical in physical_ids.items()}
    preferred_nodes = [node_indices[q] for q in (preferred or []) if q in node_indices]
    roots = list(
        dict.fromkeys(
            preferred_nodes + sorted(graph.node_indices(), key=lambda q: (-graph.degree(q), q))
        )
    )
    for root in roots:
        selected = [root]
        seen = {root}
        for node in selected:
            for neighbor in sorted(graph.neighbors(node)):
                if len(selected) == width:
                    return sorted(physical_ids[q] for q in selected)
                if neighbor not in seen:
                    seen.add(neighbor)
                    selected.append(neighbor)
        if len(selected) == width:
            return sorted(physical_ids[q] for q in selected)
    raise ValueError("QUOPS requires a connected physical subset of the requested width")


def compile_quops_circuit(
    device,
    circuit: QuantumCircuit,
    *,
    seed: int | None = None,
    physical_qubits: list[int] | None = None,
) -> CompiledQUOPSCircuit:
    """Optimize a test circuit, returning compact u/cx/cz wires and their mapping.

    With no requested IBM layout, Qiskit's default layout selection searches the
    full backend using available calibration data. As in the pyGSTi full-stack MCFE
    adapter, candidates that use routing ancillas are rejected: enlarging the
    randomized state would change the measured channel. If ten searches fail,
    route within a connected subset seeded by the last candidate.
    Pass the returned subset on subsequent calls to keep one layout per width.
    """
    if circuit.num_clbits or any(i.operation.name == "measure" for i in circuit.data):
        raise ValueError("QUOPS test compilation expects a measurement-free circuit")
    validate_quops_options(device)
    width = circuit.num_qubits
    physical = _layout(width, physical_qubits, device.num_qubits)
    common = dict(optimization_level=3, seed_transpiler=seed, num_processes=1)
    layout_method = "fixed_subset" if physical_qubits is not None else "all_to_all"
    actual_seed = seed
    constrained_local = (
        isinstance(device, LocalAerDevice) and device._backend.coupling_map is not None
    )
    if isinstance(device, QiskitBackend) or constrained_local:
        if physical_qubits is not None:
            compiled = transpile(
                circuit,
                target=_restricted_target(device._backend.target, physical),
                layout_method="sabre",
                routing_method="sabre",
                **common,
            )
            compact = _compact(compiled, list(range(width)))
        else:
            for attempt in range(10):
                kwargs = {**common, "seed_transpiler": None if seed is None else seed + attempt}
                compiled = transpile(
                    circuit,
                    backend=device._backend,
                    # Forcing SABRE layout disables Qiskit's calibration-based improvements.
                    **kwargs,
                )
                initial = compiled.layout.initial_index_layout(filter_ancillas=True)
                final = compiled.layout.final_index_layout(filter_ancillas=True)
                physical = sorted(initial)
                active = {
                    compiled.find_bit(q).index
                    for inst in compiled.data
                    if inst.operation.name != "barrier"
                    for q in inst.qubits
                }
                if active <= set(physical) and set(final) == set(physical):
                    compact = _compact(compiled, physical)
                    layout_method = "default_auxiliary_free"
                    actual_seed = kwargs["seed_transpiler"]
                    break
            else:
                physical = _connected_subset(device, width, preferred=physical)
                compiled = transpile(
                    circuit,
                    target=_restricted_target(device._backend.target, physical),
                    layout_method="sabre",
                    routing_method="sabre",
                    **common,
                )
                compact = _compact(compiled, list(range(width)))
                layout_method = "connected_subset_fallback"
    elif isinstance(device, (LocalAerDevice, BraketDevice, QuantinuumDevice)):
        if isinstance(device, BraketDevice) and not device.simulator and physical_qubits is None:
            physical = _connected_subset(device, width)
            layout_method = "connected_subset"
        basis = _braket_basis(device) if isinstance(device, BraketDevice) else ["u", "cx"]
        coupling = _coupling(device, physical) if isinstance(device, BraketDevice) else None
        compact = transpile(circuit, basis_gates=basis, coupling_map=coupling, **common)
        if isinstance(device, BraketDevice) and not device.simulator:
            compact = _braket_fixed_rx(compact)
        if isinstance(device, QuantinuumDevice):
            from pytket.extensions.qiskit import tk_to_qiskit

            compact = tk_to_qiskit(_quantinuum_native(compact))
    else:
        raise ValueError(f"QUOPS does not support device type {type(device).__name__}")
    canonical = transpile(
        compact,
        basis_gates=["u", "cx", "cz"],
        optimization_level=0,
        seed_transpiler=seed,
        num_processes=1,
    )
    packages = ["qiskit"]
    if isinstance(device, QiskitBackend):
        packages.append("qiskit-ibm-runtime")
    elif isinstance(device, LocalAerDevice):
        packages.append("qiskit-aer")
    elif isinstance(device, QuantinuumDevice):
        packages.extend(["pytket", "pytket-qiskit"])
    elif isinstance(device, BraketDevice):
        packages.append("amazon-braket-sdk")
    metadata = {
        "optimization_level": 3,
        "seed_transpiler": actual_seed,
        "layout_method": layout_method,
        "native_gates": ["ZZMax", "Rz", "PhasedX"]
        if isinstance(device, QuantinuumDevice)
        else sorted(set(compact.count_ops()) - {"barrier"}),
        "sdk_versions": {package: version(package) for package in packages},
    }
    if isinstance(device, (QiskitBackend, LocalAerDevice)):
        metadata["backend_version"] = device._backend.backend_version
        # Use already cached calibration data; metadata must not make an API call.
        cached_properties = getattr(device._backend, "_properties", None)
        calibrated_at = getattr(cached_properties, "last_update_date", None)
        metadata["calibration_updated_at"] = calibrated_at.isoformat() if calibrated_at else None
    return CompiledQUOPSCircuit(canonical, physical, metadata)


def prepare_quops_circuit(
    device, circuit: QuantumCircuit, physical_qubits: list[int]
) -> QuantumCircuit:
    """Translate an assembled mirror without changing its layout or cancelling gates."""
    physical = _layout(circuit.num_qubits, physical_qubits, device.num_qubits)
    if isinstance(device, QiskitBackend):
        return transpile(
            circuit,
            backend=device._backend,
            initial_layout=physical,
            routing_method="none",
            optimization_level=0,
            num_processes=1,
        )
    if isinstance(device, LocalAerDevice):
        layout_options = {"initial_layout": physical}
        if device._backend.coupling_map is None and physical == list(range(circuit.num_qubits)):
            layout_options = {}
        return transpile(
            circuit,
            backend=device._backend,
            **layout_options,
            routing_method="none",
            optimization_level=0,
            num_processes=1,
        )
    if isinstance(device, QuantinuumDevice):
        if physical != list(range(circuit.num_qubits)):
            raise ValueError(
                "QUOPS Quantinuum requires register indices 0..width-1; placement is device-managed"
            )
        from pytket.extensions.qiskit import tk_to_qiskit

        # Store a Qiskit representation for the common resource estimator.
        # Submission restores the native gate labels without optimizations.
        return tk_to_qiskit(_quantinuum_native(circuit))
    if isinstance(device, BraketDevice):
        native = transpile(
            circuit, basis_gates=_braket_basis(device), optimization_level=0, num_processes=1
        )
        if not device.simulator:
            native = _braket_fixed_rx(native)
        mapped = QuantumCircuit(max(physical) + 1, circuit.num_clbits)
        mapped.compose(native, qubits=physical, clbits=range(circuit.num_clbits), inplace=True)
        return mapped
    raise ValueError(f"QUOPS does not support device type {type(device).__name__}")


def _braket_circuit(circuit: QuantumCircuit, *, verbatim: bool):
    """Convert only our explicit native basis; drop barriers only inside a verbatim box."""
    from braket.circuits import Circuit

    body = Circuit()
    measurements = []
    for inst in circuit.data:
        name = inst.operation.name
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            measurements.append((circuit.find_bit(inst.clbits[0]).index, qubits[0]))
        elif name in ("rx", "ry", "rz"):
            getattr(body, name)(qubits[0], float(inst.operation.params[0]))
        elif name in ("cx", "cz"):
            getattr(body, "cnot" if name == "cx" else name)(*qubits)
        else:
            raise ValueError(f"Unsupported QUOPS Braket instruction: {name}")
    ordered = [q for _, q in sorted(measurements)]
    if not ordered or ordered != sorted(set(ordered)):
        raise ValueError("QUOPS Braket measurements must use distinct ascending physical qubits")
    result = Circuit().add_verbatim_box(body) if verbatim else body
    result.measure(ordered)
    return result


def submit_quops_circuits(
    device,
    circuits: list[QuantumCircuit],
    *,
    shots: int,
    dynamical_decoupling: bool = False,
):
    """Submit prepared circuits with no additional optimization or automatic twirling."""
    if not circuits:
        raise ValueError("QUOPS cannot submit an empty circuit batch")
    if type(shots) is not int or shots <= 0:
        raise ValueError("QUOPS shots must be a positive integer")
    validate_quops_options(device, dynamical_decoupling=dynamical_decoupling)
    if isinstance(device, QiskitBackend):
        from qiskit_ibm_runtime import SamplerOptions, SamplerV2

        options = SamplerOptions(
            dynamical_decoupling={"enable": dynamical_decoupling, "sequence_type": "XX"},
            twirling={"enable_gates": False, "enable_measure": False},
        )
        job = SamplerV2(mode=device._backend, options=options).run(circuits, shots=shots)
        return [QiskitJob(job.job_id(), job=job, device=device)]
    if isinstance(device, LocalAerDevice):
        from time import perf_counter
        from uuid import uuid4
        from metriq_gym.local.job import LocalAerJob

        started = perf_counter()
        counts = device._backend.run(circuits, shots=shots).result().get_counts()
        return [
            LocalAerJob(
                str(uuid4()), device=device, counts=counts, execution_time=perf_counter() - started
            )
        ]
    if isinstance(device, QuantinuumDevice):
        return [
            device.submit(
                [_quantinuum_native(c) for c in circuits], shots=shots, optimisation_level=0
            )
        ]
    if isinstance(device, BraketDevice):
        _braket_basis(device)  # Validate before any submission.
        converted = [_braket_circuit(c, verbatim=not device.simulator) for c in circuits]
        jobs = device.submit(converted, shots=shots, disable_qubit_rewiring=not device.simulator)
        return jobs if isinstance(jobs, list) else [jobs]
    raise ValueError(f"QUOPS does not support device type {type(device).__name__}")
