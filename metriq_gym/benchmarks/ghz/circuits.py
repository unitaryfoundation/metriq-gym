"""GHZ state preparation and verification circuits."""

import numpy as np
from qiskit import QuantumCircuit


def _dissemination_rounds(graph, root: int, num_qubits: int) -> list[list[tuple[int, int]]]:
    """Binary dissemination tree from root, as rounds of disjoint CNOT edges.

    Every qubit already holding the state copies it to one unused neighbour per
    round, so the prepared set doubles each round and the edges within a round
    act on disjoint qubits. That gives depth logarithmic in num_qubits on
    well-connected devices (a star, i.e. one control for every CNOT, is linear),
    and degrades to the old chain on a path graph where no other option exists.
    """
    prepared = [root]
    visited = {root}
    rounds: list[list[tuple[int, int]]] = []

    while len(visited) < num_qubits:
        round_edges: list[tuple[int, int]] = []
        # Snapshot: qubits prepared in this round only spread on the next one.
        for node in list(prepared):
            if len(visited) >= num_qubits:
                break
            # Sorted: rustworkx does not guarantee a stable neighbour order, so
            # without this the selected data qubits (and therefore the whole
            # layout) can differ between identical runs.
            for neighbor in sorted(graph.neighbors(node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    prepared.append(neighbor)
                    round_edges.append((node, neighbor))
                    break
        if not round_edges:
            break
        rounds.append(round_edges)

    return rounds


def _bfs_edges(graph, root: int, num_qubits: int) -> list[tuple[int, int]]:
    """Flattened dissemination-tree edges from root, limited to num_qubits nodes."""
    edges: list[tuple[int, int]] = []
    for round_edges in _dissemination_rounds(graph, root, num_qubits):
        edges.extend(round_edges)

    return edges


def sample_ghz_stabilizers(
    n: int, num_stabilizers: int, seed: int | None = None
) -> list[list[int]]:
    """Uniformly sample off-diagonal GHZ stabilizers for DFE.

    The off-diagonal half of the GHZ stabilizer group is {X^(x)N·Z_s} with s
    ranging over the 2^(n-1) even-weight bitstrings. As a Pauli,
    X^(x)N·Z_s = (-1)^(|s|/2) times a tensor with Y inside the support of s
    and X outside it. Each sample is returned as the sorted list of data-qubit
    positions in s, i.e. the qubits to measure in the Y basis.

    Uniformity: a uniform draw over all 2^n strings, with one uniformly chosen
    bit flipped when the weight is odd, lands on every even-weight string with
    equal probability.
    """
    rng = np.random.default_rng(seed)
    samples: list[list[int]] = []
    for _ in range(num_stabilizers):
        bits = rng.integers(0, 2, size=n)
        if int(bits.sum()) % 2:
            bits[int(rng.integers(0, n))] ^= 1
        samples.append([int(i) for i in np.flatnonzero(bits)])
    return samples


def build_ghz_circuits(
    graph,
    num_qubits: int,
    method: str = "dfe",
    phases: list[float] | None = None,
    stabilizers: list[list[int]] | None = None,
) -> tuple[list[QuantumCircuit], list[int]]:
    """Build GHZ verification circuits using BFS on the device connectivity graph.

    Returns:
        (circuits, data_qubits) where circuits is the list of verification
        circuits and data_qubits are the physical qubit indices used.

    Raises:
        ValueError: if the device has fewer than `num_qubits` qubits or if a
            BFS from the root cannot reach `num_qubits` distinct qubits
            (e.g. disconnected connectivity graph).
    """
    total_device_qubits = graph.num_nodes()
    if num_qubits > total_device_qubits:
        raise ValueError(
            f"Requested {num_qubits} qubits but device only exposes "
            f"{total_device_qubits} qubits in its connectivity graph"
        )

    # BFS from node 0 to select data qubits
    root = 0
    bfs = _bfs_edges(graph, root, num_qubits)
    data_qubits_set = {root}
    for ctrl, targ in bfs:
        data_qubits_set.add(ctrl)
        data_qubits_set.add(targ)
    data_qubits = sorted(data_qubits_set)[:num_qubits]
    if len(data_qubits) < num_qubits:
        raise ValueError(
            f"BFS from root {root} could only reach {len(data_qubits)} of the "
            f"{num_qubits} requested qubits; the device connectivity graph may "
            f"be disconnected"
        )

    # Size the register to the highest physical index in use rather than the
    # whole device: qubit i still maps to physical qubit i (so the BFS layout
    # is preserved on devices that pin by index), but backends that simulate
    # every declared wire (e.g. the Quantinuum emulators) no longer pay for
    # idle qubits. On an all-to-all device this is exactly num_qubits wide.
    n_total = max(data_qubits) + 1

    def _make_ghz_circuit() -> QuantumCircuit:
        """Create base GHZ state preparation circuit."""
        qc = QuantumCircuit(n_total, len(data_qubits))
        # Hadamard on root
        qc.h(data_qubits[0])
        # CNOT dissemination tree
        for ctrl, targ in bfs:
            qc.cx(ctrl, targ)
        # Barrier only the data qubits: a full-width barrier marks every
        # wire as used, so backends that drop idle wires (e.g. the Quantinuum
        # transform) would otherwise submit a device-width circuit.
        qc.barrier(data_qubits)
        return qc

    def _add_measurements(qc: QuantumCircuit) -> None:
        """Add measurement gates for the data qubits."""
        qc.measure(data_qubits, list(range(len(data_qubits))))

    if method == "two_setting_bound":
        # Z-basis circuit
        z_qc = _make_ghz_circuit()
        _add_measurements(z_qc)

        # X-basis circuit
        x_qc = _make_ghz_circuit()
        for dq in data_qubits:
            x_qc.h(dq)
        _add_measurements(x_qc)

        return [z_qc, x_qc], data_qubits

    elif method == "dfe":
        if stabilizers is None:
            raise ValueError("stabilizers required for dfe method")

        # Z-basis circuit: covers the diagonal half of the stabilizer group.
        z_qc = _make_ghz_circuit()
        _add_measurements(z_qc)

        # One circuit per sampled off-diagonal stabilizer X^(x)N·Z_s: qubits
        # in the support of s are measured in the Y basis (Sdg; H; measure Z),
        # the rest in X (H; measure Z).
        stab_circuits = []
        for y_positions in stabilizers:
            y_set = set(y_positions)
            stab_qc = _make_ghz_circuit()
            for i, dq in enumerate(data_qubits):
                if i in y_set:
                    stab_qc.sdg(dq)
                stab_qc.h(dq)
            _add_measurements(stab_qc)
            stab_circuits.append(stab_qc)

        return [z_qc] + stab_circuits, data_qubits

    elif method in ("parity_oscillation", "compressed_sensing"):
        if phases is None:
            raise ValueError(f"phases required for {method} method")

        # Z-basis circuit
        z_qc = _make_ghz_circuit()
        _add_measurements(z_qc)

        # Oscillation circuits for each phase. Both methods share the same
        # per-circuit structure (Rz(phi); H; measure in Z); the only difference
        # is the phase grid chosen by the caller: a uniform grid over the full
        # period [0, 2π) for parity_oscillation, random phases on [0, 2π) for
        # compressed_sensing.
        osc_circuits = []
        for phi in phases:
            osc_qc = _make_ghz_circuit()
            for dq in data_qubits:
                osc_qc.rz(phi, dq)
                osc_qc.h(dq)
            _add_measurements(osc_qc)
            osc_circuits.append(osc_qc)

        return [z_qc] + osc_circuits, data_qubits

    else:
        raise ValueError(f"Unknown verification method: {method}")
