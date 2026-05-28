"""Mermin-Peres magic square game benchmark.

Summary:
    Two-party benchmark for the Mermin-Peres magic square game on a
    pair of shared Bell states. Alice receives a uniformly random row
    index r in {1, 2, 3}, Bob receives a uniformly random column index
    c in {1, 2, 3}; they win iff their reported values at the shared
    cell (r, c) agree, subject to row-product and column-product
    constraints of the magic square. The optimal classical strategy
    wins with probability 8/9; the quantum strategy on two shared Bell
    pairs wins with probability 1.

Hilbert space and state:
    Alice holds two qubits and Bob holds two qubits, totalling four.
    The unit vector is the tensor product of two Bell pairs distributed
    across the parties: |Phi+>_{a1 b1} (x) |Phi+>_{a2 b2}, equivalently
    the maximally entangled state of two 4-dimensional systems
    (1/2) sum_{j=0}^{3} |j>_A |j>_B.

Strategy:
    Standard Mermin-Peres assignment of nine commuting two-qubit Pauli
    operators to the 3x3 grid:

        Row 1: I (x) X    X (x) I    X (x) X
        Row 2: Y (x) I    I (x) Y    Y (x) Y
        Row 3: Y (x) X    X (x) Y    Z (x) Z

    All row products are +I, columns 1 and 2 multiply to +I, column 3
    multiplies to -I. Alice measures the three commuting operators in
    row r simultaneously on her two qubits; Bob measures the three
    commuting operators in column c on his two qubits. Each two-qubit
    measurement is a Z-basis readout after a row- or column-specific
    basis-change unitary.

    On the two-Bell-pair state, single-qubit X and Z operators are
    correlated across Alice and Bob while single-qubit Y operators are
    anti-correlated. The shared-cell operator at (r, c) therefore
    produces outcomes that agree up to a sign sigma(r, c) given by the
    parity of the number of Y's in the operator. Applying this sign
    correction to Bob's reported value preserves the column-product
    constraint and yields perfect Alice/Bob agreement at the shared
    cell.

Result interpretation:
    Polling returns MagicSquareResult with:
        - win_probability: empirical win rate, averaged over the nine
          uniformly distributed (r, c) inputs. Each input is sampled
          ``shots`` times. Equal to 1 under ideal execution.
        - achievement_ratio: (win_probability - 8/9) / (1 - 8/9) =
          9 * win_probability - 8. Positive values indicate strictly
          super-classical performance.
        - violated: bool, true iff win_probability > 8/9.

References:
    - Mermin, "Simple unified form for the major no-hidden-variables
      theorems", *Phys. Rev. Lett.* 65, 3373 (1990).
    - Peres, "Incompatible results of quantum measurements",
      *Phys. Lett. A* 151, 107 (1990).
"""

from dataclasses import dataclass, field

from qbraid import GateModelResultData, QuantumDevice, QuantumJob
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.resource_estimation import CircuitBatch


CLASSICAL_BOUND = 8.0 / 9.0
QUANTUM_BOUND = 1.0


# Sign correction at the shared cell (r, c) for the two-Bell-pair state.
# Equals +1 when the operator at (r, c) has an even number of Y's
# (correlated) and -1 otherwise (anti-correlated).
SHARED_CELL_SIGN: dict[tuple[int, int], int] = {
    (1, 1): +1,  # I (x) X
    (1, 2): +1,  # X (x) I
    (1, 3): +1,  # X (x) X
    (2, 1): -1,  # Y (x) I
    (2, 2): -1,  # I (x) Y
    (2, 3): +1,  # Y (x) Y
    (3, 1): -1,  # Y (x) X
    (3, 2): -1,  # X (x) Y
    (3, 3): +1,  # Z (x) Z
}


def prepare_two_bell_pairs(
    num_qubits: int, alice_qubits: tuple[int, int], bob_qubits: tuple[int, int]
) -> QuantumCircuit:
    """Prepare |Phi+>_(a1 b1) (x) |Phi+>_(a2 b2) across the chosen four qubits."""
    a1, a2 = alice_qubits
    b1, b2 = bob_qubits
    qc = QuantumCircuit(num_qubits, name="2BellPairs")
    qc.h(a1)
    qc.cx(a1, b1)
    qc.h(a2)
    qc.cx(a2, b2)
    return qc


def _x_basis_change(qc: QuantumCircuit, q: int) -> None:
    """Append the basis change so the subsequent Z-measure implements X."""
    qc.h(q)


def _y_basis_change(qc: QuantumCircuit, q: int) -> None:
    """Append the basis change so the subsequent Z-measure implements Y."""
    qc.sdg(q)
    qc.h(q)


def _bell_basis_change(qc: QuantumCircuit, q_first: int, q_second: int) -> None:
    """Diagonalise {X (x) X, Y (x) Y, Z (x) Z} via CNOT + H on the first qubit."""
    qc.cx(q_first, q_second)
    qc.h(q_first)


def _row3_basis_change(qc: QuantumCircuit, q_first: int, q_second: int) -> None:
    """Diagonalise {Y (x) X, X (x) Y, Z (x) Z} via Sdg on first qubit + Bell basis change.

    Direct verification: Sdg on the first qubit maps the simultaneous
    eigenstates (|00> +/- i|11>)/sqrt(2), (|01> +/- i|10>)/sqrt(2) to the
    Bell states |Phi+>, |Phi->, |Psi+>, |Psi->, which the subsequent
    Bell-basis change then routes to computational-basis states.
    """
    qc.sdg(q_first)
    _bell_basis_change(qc, q_first, q_second)


def _alice_row_basis_change(qc: QuantumCircuit, r: int, a1: int, a2: int) -> None:
    if r == 1:
        _x_basis_change(qc, a1)
        _x_basis_change(qc, a2)
    elif r == 2:
        _y_basis_change(qc, a1)
        _y_basis_change(qc, a2)
    elif r == 3:
        _row3_basis_change(qc, a1, a2)
    else:
        raise ValueError(f"row r must be in {{1, 2, 3}}, got {r}")


def _bob_col_basis_change(qc: QuantumCircuit, c: int, b1: int, b2: int) -> None:
    if c == 1:
        # Col 1 ops: I (x) X (X on q_second), Y (x) I (Y on q_first), Y (x) X
        _y_basis_change(qc, b1)
        _x_basis_change(qc, b2)
    elif c == 2:
        # Col 2 ops: X (x) I (X on q_first), I (x) Y (Y on q_second), X (x) Y
        _x_basis_change(qc, b1)
        _y_basis_change(qc, b2)
    elif c == 3:
        # Col 3 ops: X (x) X, Y (x) Y, Z (x) Z
        _bell_basis_change(qc, b1, b2)
    else:
        raise ValueError(f"col c must be in {{1, 2, 3}}, got {c}")


def _decode_alice_row(r: int, b0: int, b1: int) -> tuple[int, int, int]:
    """Decode (b0, b1) -> (a_1, a_2, a_3), the values at row r's three cells.

    Returns the +/- 1 eigenvalues for the three operators in row r, in
    column order. The decoders follow from the operator commutation
    structure (each row's product is +I, so the third value is the
    product of the other two for rows 1 and 2; row 3 also has Z (x) Z
    in the third column).
    """
    v0 = 1 if b0 == 0 else -1
    v1 = 1 if b1 == 0 else -1
    if r == 1:
        # Row 1 ops in col order: I (x) X, X (x) I, X (x) X. Bits b0, b1
        # are eigenvalues of X on q_first, X on q_second.
        return (v1, v0, v0 * v1)
    if r == 2:
        # Row 2 ops: Y (x) I, I (x) Y, Y (x) Y. Bits b0, b1 are
        # eigenvalues of Y on q_first, Y on q_second.
        return (v0, v1, v0 * v1)
    if r == 3:
        # Row 3 ops: Y (x) X, X (x) Y, Z (x) Z. After Sdg_a1 + Bell
        # basis change, the four simultaneous eigenstates map to comp
        # basis with b0 encoding Y (x) X, b1 encoding Z (x) Z, and
        # X (x) Y = (Y (x) X) * (Z (x) Z) following from the operator
        # product (YX)(XY)(ZZ) = +I.
        return (v0, v0 * v1, v1)
    raise ValueError(f"row r must be in {{1, 2, 3}}, got {r}")


def _decode_bob_col(c: int, b0: int, b1: int) -> tuple[int, int, int]:
    """Decode (b0, b1) -> (b_1, b_2, b_3), the values at column c's three cells."""
    v0 = 1 if b0 == 0 else -1
    v1 = 1 if b1 == 0 else -1
    if c == 1:
        # Col 1 ops in row order: I (x) X (row 1), Y (x) I (row 2),
        # Y (x) X (row 3). With Y on q_first measured (b0) and X on
        # q_second (b1).
        return (v1, v0, v0 * v1)
    if c == 2:
        # Col 2 ops: X (x) I (row 1), I (x) Y (row 2), X (x) Y (row 3).
        return (v0, v1, v0 * v1)
    if c == 3:
        # Col 3 ops: X (x) X (row 1), Y (x) Y (row 2), Z (x) Z (row 3).
        # After Bell basis change, b0 encodes X (x) X, b1 encodes
        # Z (x) Z, and Y (x) Y picks up a sign from
        # (XX)(YY)(ZZ) = -I so e_YY = -e_XX * e_ZZ.
        return (v0, -v0 * v1, v1)
    raise ValueError(f"col c must be in {{1, 2, 3}}, got {c}")


def build_magic_square_circuits(
    alice_qubits: tuple[int, int],
    bob_qubits: tuple[int, int],
    num_qubits: int,
) -> list[QuantumCircuit]:
    """Build the nine measurement circuits, indexed by (r, c) in row-major order."""
    a1, a2 = alice_qubits
    b1, b2 = bob_qubits
    circuits: list[QuantumCircuit] = []
    for r in (1, 2, 3):
        for c in (1, 2, 3):
            qc = QuantumCircuit(num_qubits, 4, name=f"MagicSquare_r{r}_c{c}")
            qc.compose(
                prepare_two_bell_pairs(num_qubits, alice_qubits, bob_qubits), inplace=True
            )
            _alice_row_basis_change(qc, r, a1, a2)
            _bob_col_basis_change(qc, c, b1, b2)
            qc.measure(a1, 0)
            qc.measure(a2, 1)
            qc.measure(b1, 2)
            qc.measure(b2, 3)
            circuits.append(qc)
    return circuits


def win_probability_from_counts(counts: dict[str, int], r: int, c: int) -> float:
    """Empirical win probability of one (r, c) circuit's outcomes."""
    total = 0
    wins = 0
    sigma = SHARED_CELL_SIGN[(r, c)]
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        # Qiskit convention: cleaned[-1] is classical bit 0.
        bits = [int(cleaned[-1 - i]) for i in range(4)]
        alice_row = _decode_alice_row(r, bits[0], bits[1])
        bob_col = _decode_bob_col(c, bits[2], bits[3])
        a_c = alice_row[c - 1]
        b_r = bob_col[r - 1]
        if a_c == sigma * b_r:
            wins += count
        total += count
    if total == 0:
        return 0.0
    return wins / total


class MagicSquareResult(BenchmarkResult):
    win_probability: float
    achievement_ratio: float
    violated: bool

    def _iter_metric_items(self):
        yield "win_probability", float(self.win_probability), None
        yield "achievement_ratio", float(self.achievement_ratio), None
        yield "violated", float(self.violated), None

    def compute_score(self) -> BenchmarkScore:
        return BenchmarkScore(value=float(self.win_probability))


@dataclass
class MagicSquareData(BenchmarkData):
    shots: int = 0
    alice_qubits: list[int] = field(default_factory=lambda: [0, 1])
    bob_qubits: list[int] = field(default_factory=lambda: [2, 3])
    num_qubits: int = 0


class MagicSquare(Benchmark):
    """Benchmark class for the Mermin-Peres magic square game."""

    def _qubits(self) -> tuple[tuple[int, int], tuple[int, int]]:
        alice = list(self.params.alice_qubits)
        bob = list(self.params.bob_qubits)
        if len(alice) != 2 or len(bob) != 2:
            raise ValueError(
                f"alice_qubits and bob_qubits must each contain 2 entries; got alice={alice}, bob={bob}"
            )
        if len(set(alice + bob)) != 4:
            raise ValueError(
                f"alice_qubits and bob_qubits must be four distinct qubits; got {alice + bob}"
            )
        return (alice[0], alice[1]), (bob[0], bob[1])

    def dispatch_handler(self, device: QuantumDevice) -> MagicSquareData:
        shots = self.params.shots
        alice_qubits, bob_qubits = self._qubits()
        num_qubits = device.num_qubits
        circuits = build_magic_square_circuits(alice_qubits, bob_qubits, num_qubits)
        return MagicSquareData.from_quantum_job(
            device.run(circuits, shots=shots),
            shots=shots,
            alice_qubits=list(alice_qubits),
            bob_qubits=list(bob_qubits),
            num_qubits=num_qubits,
        )

    def poll_handler(
        self,
        job_data: MagicSquareData,
        result_data: list[GateModelResultData],
        quantum_jobs: list[QuantumJob],
    ) -> MagicSquareResult:
        counts_list = flatten_counts(result_data)
        if len(counts_list) != 9:
            raise ValueError(f"expected 9 count dicts, got {len(counts_list)}")
        win_probs: list[float] = []
        idx = 0
        for r in (1, 2, 3):
            for c in (1, 2, 3):
                win_probs.append(win_probability_from_counts(counts_list[idx], r, c))
                idx += 1
        avg = sum(win_probs) / len(win_probs)
        achievement_ratio = (avg - CLASSICAL_BOUND) / (QUANTUM_BOUND - CLASSICAL_BOUND)
        return MagicSquareResult(
            win_probability=avg,
            achievement_ratio=achievement_ratio,
            violated=avg > CLASSICAL_BOUND,
        )

    def estimate_resources_handler(
        self,
        device: QuantumDevice,
    ) -> list[CircuitBatch]:
        alice_qubits, bob_qubits = self._qubits()
        circuits = build_magic_square_circuits(alice_qubits, bob_qubits, device.num_qubits)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]


