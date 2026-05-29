import rustworkx as rx

from metriq_gym.benchmarks.magic_square_lccs import (
    build_batched_circuits,
    color_plaquettes,
    compute_lccs,
    enumerate_4cycles,
)


def _square_lattice(rows: int, cols: int) -> rx.PyGraph:
    """Build an ``rows x cols`` square-lattice PyGraph with row-major node ids."""
    g = rx.PyGraph()
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    idx = {node: i for i, node in enumerate(nodes)}
    g.add_nodes_from(nodes)
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                g.add_edge(idx[(r, c)], idx[(r, c + 1)], None)
            if r + 1 < rows:
                g.add_edge(idx[(r, c)], idx[(r + 1, c)], None)
    return g


def test_enumerate_4cycles_on_3x3_square_lattice():
    # A 3x3 grid has 4 plaquettes.
    g = _square_lattice(3, 3)
    cycles = enumerate_4cycles(g)
    assert len(cycles) == 4
    # Each cycle should have 4 distinct qubits and all 4 magic-square edges.
    for a1, a2, b1, b2 in cycles:
        assert len({a1, a2, b1, b2}) == 4
        assert g.has_edge(a1, b1)
        assert g.has_edge(a2, b2)
        assert g.has_edge(a1, a2)
        assert g.has_edge(b1, b2)


def test_enumerate_4cycles_on_5x5_square_lattice():
    # (N-1) x (M-1) plaquettes on an N x M square lattice.
    g = _square_lattice(5, 5)
    cycles = enumerate_4cycles(g)
    assert len(cycles) == 16


def test_enumerate_4cycles_on_heavy_hex_like_graph_returns_none():
    # Hexagonal cycles have no 4-cycle subgraphs.
    g = rx.PyGraph()
    g.add_nodes_from(list(range(6)))
    for i in range(6):
        g.add_edge(i, (i + 1) % 6, None)
    cycles = enumerate_4cycles(g)
    assert cycles == []


def test_enumerate_4cycles_canonicalisation_dedupes_rotations():
    # K_4: every 3-cycle is closed, has C(4, 2) / 2 = 3 distinct 4-cycles
    # (each 4-cycle = a Hamiltonian cycle of K_4).
    g = rx.PyGraph()
    g.add_nodes_from(list(range(4)))
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j, None)
    cycles = enumerate_4cycles(g)
    # The K_4 vertex set is the same for every Hamiltonian 4-cycle; the
    # current enumerator dedupes by vertex set, so it reports a single
    # canonical labeling for K_4. Either 1 or 3 is acceptable depending
    # on the dedup semantics; for our magic-square purposes the qubits
    # being shared makes them logically the same site.
    assert len(cycles) == 1


def test_compute_lccs_empty_passing_returns_zero():
    g = _square_lattice(3, 3)
    assert compute_lccs([], g) == 0


def test_compute_lccs_single_passing_cycle_covers_four_qubits():
    g = _square_lattice(3, 3)
    cycles = enumerate_4cycles(g)
    # Any single cycle covers 4 qubits, all connected in g.
    assert compute_lccs([cycles[0]], g) == 4


def test_compute_lccs_all_passing_covers_full_lattice():
    g = _square_lattice(3, 3)
    cycles = enumerate_4cycles(g)
    # All 4 plaquettes cover all 9 qubits, all connected.
    assert compute_lccs(cycles, g) == 9


def test_color_plaquettes_square_lattice_uses_four_colors():
    # On any NxN square lattice with N >= 3, the optimal plaquette
    # coloring uses 4 colors (2x2 block coloring by parity of plaquette
    # coordinates). The greedy heuristic finds this.
    for n in (3, 4, 5, 6):
        g = _square_lattice(n, n)
        cycles = enumerate_4cycles(g)
        groups = color_plaquettes(cycles)
        # 4-color upper bound (and exact for n >= 3).
        assert 1 <= len(groups) <= 4
        # Vertex-disjoint within each group.
        for group in groups:
            seen: set[int] = set()
            for plaq in group:
                pq = set(plaq)
                assert seen.isdisjoint(pq)
                seen.update(pq)


def test_color_plaquettes_preserves_total_count():
    g = _square_lattice(5, 5)
    cycles = enumerate_4cycles(g)
    groups = color_plaquettes(cycles)
    assert sum(len(group) for group in groups) == len(cycles)


def test_build_batched_circuits_count_and_classical_bits():
    # 4x4 square lattice has 9 plaquettes; the 4-coloring assigns
    # specific counts per color group.
    g = _square_lattice(4, 4)
    cycles = enumerate_4cycles(g)
    groups = color_plaquettes(cycles)
    circuits = build_batched_circuits(groups, num_qubits=16)
    # 9 (r, c) inputs per color group.
    assert len(circuits) == 9 * len(groups)
    # Each circuit on group g uses 4 * len(group) classical bits.
    for group_idx, group in enumerate(groups):
        for input_idx in range(9):
            qc = circuits[9 * group_idx + input_idx]
            assert qc.num_qubits == 16
            assert qc.num_clbits == 4 * len(group)


def test_batched_circuit_count_independent_of_lattice_size():
    # On any square lattice with N >= 3, the batched circuit count is
    # 9 * 4 = 36 because the 4-coloring is optimal regardless of N.
    for n in (3, 4, 5, 6, 8):
        g = _square_lattice(n, n)
        cycles = enumerate_4cycles(g)
        groups = color_plaquettes(cycles)
        circuits = build_batched_circuits(groups, num_qubits=n * n)
        assert len(circuits) == 9 * len(groups)
        assert len(groups) == 4


def test_compute_lccs_disconnected_components():
    # Two disjoint plaquettes -> LCCS = 4.
    g = rx.PyGraph()
    g.add_nodes_from(list(range(8)))
    for i in range(0, 4):
        g.add_edge(i, (i + 1) % 4, None)
    for i in range(4, 8):
        g.add_edge(i, ((i - 4 + 1) % 4) + 4, None)
    cycles = enumerate_4cycles(g)
    assert len(cycles) == 2
    assert compute_lccs(cycles, g) == 4
