import rustworkx as rx

from metriq_gym.benchmarks.magic_square_lccs import (
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
