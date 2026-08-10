"""GHZ state preparation and verification benchmark.

Summary:
    Constructs a GHZ state on the device using a binary dissemination tree over
    the device's connectivity graph (logarithmic depth where connectivity
    allows), then estimates fidelity via one of three verification methods:
    direct fidelity estimation (DFE), parity oscillation curve fitting, or
    compressed sensing (parity samples at random phases, sparse spectrum
    recovery by L1 minimization).

    Two modes:
    - Fixed size (`num_qubits`): verify a GHZ state of one requested size.
    - Size search (`size_search: true`): build GHZ states at halving sizes
      from the device size down to `min_qubits` (e.g. 156, 78, 39, 19, 9),
      all dispatched in a single job, and report the largest size whose
      fidelity clears `fidelity_threshold` by `confidence_sigma` standard
      deviations. Defaults to compressed sensing: DFE bounds the non-GHZ
      parity contribution from the observed Z distribution, and that bound
      stops holding once 2^N outruns the shot count, which the larger searched
      sizes always do.

Result interpretation:
    - population: probability of measuring all-zero or all-one bitstrings (Z basis).
    - coherence: off-diagonal element magnitude. DFE takes the global X parity
      and subtracts the largest contribution the non-GHZ complementary pairs
      could have made, since X^(x)N connects every bitstring to its complement
      and not only 0...0 to 1...1; without that subtraction a product state
      such as |+>^(x)N reports coherence 1. The oscillation methods instead
      isolate the N-qubit Fourier component, by curve fit or by LASSO over
      random-phase parity samples.
    - fidelity: (population + coherence) / 2.
    - recovered_frequency (compressed sensing only): dominant frequency of the
      parity signal, which equals N only if the full N-qubit GHZ state was
      actually prepared; a partially entangled state shows a lower frequency.
    - largest_passing_size / device_fraction (size search only): the largest
      searched size with fidelity above the threshold, and that size divided
      by the device size. Per-size fidelities are reported in search_sizes /
      search_fidelities so no resolution is lost to the threshold.

References:
    - Moses et al., "A Race-Track Trapped-Ion Quantum Processor",
      Phys. Rev. X 13, 041052 (2023). [arXiv:2305.03828]
    - "Compressed sensing verification of large entangled states",
      arXiv:2604.27824.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit
from typing import TYPE_CHECKING

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.qplatform.device import connectivity_graph
from metriq_gym.resource_estimation import CircuitBatch, count_two_qubit_gates

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data / Result classes
# ---------------------------------------------------------------------------


@dataclass
class GHZData(BenchmarkData):
    num_qubits: int = 0
    method: str = "dfe"
    phases: list[float] = field(default_factory=list)
    num_flag_qubits: int = 0
    # Size-search mode: the halving size grid, how many circuits belong to
    # each size (in dispatch order), the per-size phase grids, and the
    # fidelity threshold a size must clear. Empty lists mean fixed-size mode.
    search_sizes: list[int] = field(default_factory=list)
    search_circuit_counts: list[int] = field(default_factory=list)
    search_flag_counts: list[int] = field(default_factory=list)
    confidence_sigma: float = 1.0
    search_phases: list[list[float]] = field(default_factory=list)
    fidelity_threshold: float = 0.5


class GHZResult(BenchmarkResult):
    population: BenchmarkScore = Field(description="Z-basis population of |0...0> + |1...1>")
    coherence: BenchmarkScore = Field(description="Off-diagonal coherence (X-basis parity or oscillation amplitude)")
    fidelity: BenchmarkScore = Field(description="GHZ state fidelity lower bound: (population + coherence) / 2")
    recovered_frequency: int | None = Field(
        default=None,
        description=(
            "Dominant parity-oscillation frequency recovered by compressed sensing; "
            "equals num_qubits only if the full GHZ state was prepared. "
            "None for other methods."
        ),
    )
    largest_passing_size: BenchmarkScore | None = Field(
        default=None,
        description=(
            "Size search only: largest searched GHZ size whose fidelity clears "
            "the threshold (0 when no size passes)."
        ),
    )
    device_fraction: BenchmarkScore | None = Field(
        default=None,
        description="Size search only: largest_passing_size divided by the device size.",
    )
    search_sizes: list[int] | None = Field(
        default=None,
        description="Size search only: the searched sizes, largest first.",
    )
    search_fidelities: list[float] | None = Field(
        default=None,
        description="Size search only: fidelity per searched size, aligned with search_sizes.",
    )
    search_uncertainties: list[float] | None = Field(
        default=None,
        description="Size search only: fidelity uncertainty per searched size.",
    )

    def compute_score(self) -> BenchmarkScore:
        # In size-search mode the headline number is the largest passing size,
        # matching the shape of BSEQ's largest_connected_size.
        if self.largest_passing_size is not None:
            return self.largest_passing_size
        return self.fidelity


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------


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


def _select_flag_qubits(
    graph, data_qubits: set[int], num_flags: int
) -> list[tuple[int, tuple[int, int]]]:
    """Select flag qubits together with the two data qubits each one checks.

    A ZZ stabilizer check needs exactly two data controls: the flag then records
    the parity of that pair, which is 0 for every GHZ basis state. A flag with a
    single data neighbour would instead copy the logical value and dephase the
    state, so candidates with fewer than two data neighbours are skipped.

    Returns (flag, (control_a, control_b)) pairs.
    """
    if num_flags <= 0:
        return []
    selected: list[tuple[int, tuple[int, int]]] = []
    candidates = set()
    for dq in data_qubits:
        for neighbor in sorted(graph.neighbors(dq)):
            if neighbor not in data_qubits:
                candidates.add(neighbor)
    for flag in sorted(candidates):
        if len(selected) >= num_flags:
            break
        data_neighbors = sorted(n for n in graph.neighbors(flag) if n in data_qubits)
        if len(data_neighbors) < 2:
            continue
        selected.append((flag, (data_neighbors[0], data_neighbors[1])))
    return selected


def build_ghz_circuits(
    graph,
    num_qubits: int,
    method: str = "dfe",
    phases: list[float] | None = None,
    num_flag_qubits: int = 0,
) -> tuple[list[QuantumCircuit], list[int], list[int]]:
    """Build GHZ verification circuits using BFS on the device connectivity graph.

    Returns:
        (circuits, data_qubits, flag_qubits) where circuits is the list of
        verification circuits and data/flag_qubits are the physical qubit indices used.

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

    # Select flag qubits, each with the two data qubits whose ZZ parity it checks.
    flag_checks = _select_flag_qubits(graph, set(data_qubits), num_flag_qubits)
    flag_qubits = [flag for flag, _ in flag_checks]

    n_total = total_device_qubits
    n_clbits = len(data_qubits) + len(flag_qubits)

    def _make_ghz_circuit() -> QuantumCircuit:
        """Create base GHZ state preparation circuit."""
        qc = QuantumCircuit(n_total, n_clbits)
        # Hadamard on root
        qc.h(data_qubits[0])
        # CNOT dissemination tree
        for ctrl, targ in bfs:
            qc.cx(ctrl, targ)
        qc.barrier()
        # ZZ stabilizer checks: exactly two data controls per flag.
        for flag, (ctrl_a, ctrl_b) in flag_checks:
            qc.cx(ctrl_a, flag)
            qc.cx(ctrl_b, flag)
        return qc

    def _add_measurements(qc: QuantumCircuit) -> None:
        """Add measurement gates for data and flag qubits."""
        qc.measure(data_qubits, list(range(len(data_qubits))))
        if flag_qubits:
            qc.measure(flag_qubits, list(range(len(data_qubits), n_clbits)))

    if method == "dfe":
        # Z-basis circuit
        z_qc = _make_ghz_circuit()
        _add_measurements(z_qc)

        # X-basis circuit
        x_qc = _make_ghz_circuit()
        for dq in data_qubits:
            x_qc.h(dq)
        _add_measurements(x_qc)

        return [z_qc, x_qc], data_qubits, flag_qubits

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

        return [z_qc] + osc_circuits, data_qubits, flag_qubits

    else:
        raise ValueError(f"Unknown verification method: {method}")


# ---------------------------------------------------------------------------
# Post-selection and fidelity estimation
# ---------------------------------------------------------------------------


def post_select_results(counts: dict[str, int], num_flag_qubits: int) -> dict[str, int]:
    """Filter measurement results, keeping only outcomes where all flag qubits measure 0."""
    if num_flag_qubits == 0:
        return counts

    post_selected: dict[str, int] = {}
    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        flags = cleaned[:num_flag_qubits]
        data = cleaned[num_flag_qubits:]
        if flags == "0" * num_flag_qubits:
            post_selected[data] = post_selected.get(data, 0) + count
    return post_selected


def complementary_pair_bound(z_ps: dict[str, int], n: int, total_z: int) -> float:
    """Bound the non-GHZ contribution to the global X parity.

    ``X^{\\otimes N}`` connects every bitstring to its complement, so its
    expectation is ``2 * sum_pairs Re(rho[x, x_bar])`` over all complementary
    pairs, not just the GHZ pair ``{0...0, 1...1}``. Positivity of rho gives
    ``|Re(rho[x, x_bar])| <= sqrt(P(x) P(x_bar))``, so this returns

        sum_{pairs != GHZ pair} sqrt(P(x) P(x_bar))

    estimated from the Z-basis distribution. Subtracting twice this from the
    measured parity leaves a valid lower bound on the GHZ coherence.

    The estimate is only trustworthy when the Z distribution is adequately
    sampled; see :func:`dfe_sampling_is_adequate`.
    """
    zero, one = "0" * n, "1" * n
    seen: set[str] = set()
    bound = 0.0
    for bitstring, count in z_ps.items():
        if bitstring in (zero, one) or bitstring in seen:
            continue
        complement = bitstring.translate(str.maketrans("01", "10"))
        seen.add(bitstring)
        seen.add(complement)
        p = count / total_z
        p_comp = z_ps.get(complement, 0) / total_z
        bound += math.sqrt(p * p_comp)
    return bound


def dfe_sampling_is_adequate(n: int, shots: int) -> bool:
    """Whether the Z distribution can support the complementary-pair bound.

    The bound in :func:`complementary_pair_bound` is estimated from observed
    Z-basis frequencies. Once 2^n greatly exceeds the shot count, complementary
    pairs are essentially never both observed, the bound collapses toward zero
    and stops being a valid lower bound. Require at least one expected shot per
    computational basis state.
    """
    return n <= 30 and (2**n) <= shots


def estimate_fidelity_dfe(
    z_counts: dict[str, int],
    x_counts: dict[str, int],
    n: int,
    num_flag_qubits: int,
) -> tuple[float, float, float, float]:
    """Estimate GHZ fidelity using direct fidelity estimation.

    The coherence is the global X parity corrected by the largest possible
    contribution from non-GHZ complementary pairs, so a separable state such
    as ``|+>^{\\otimes N}`` (parity 1, no entanglement) is not certified.

    Returns: (population, coherence, population_err, coherence_err)
    """
    z_ps = post_select_results(z_counts, num_flag_qubits)
    x_ps = post_select_results(x_counts, num_flag_qubits)

    total_z = sum(z_ps.values())
    total_x = sum(x_ps.values())

    if total_z == 0 or total_x == 0:
        return 0.0, 0.0, 0.0, 0.0

    population = (z_ps.get("0" * n, 0) + z_ps.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    # Magnitude of the X-basis parity. The signed value distinguishes GHZ+ from
    # GHZ-, but for a fidelity lower bound of the form
    # (population + coherence) / 2 we want the off-diagonal magnitude so that
    # GHZ-like states with a relative phase are not penalized.
    even_x = sum(c for b, c in x_ps.items() if b.count("1") % 2 == 0)
    raw_parity = abs((2 * even_x - total_x) / total_x)

    # Subtract the maximum the other complementary pairs could have
    # contributed. Without this, |+>^N reports coherence 1 despite carrying no
    # entanglement at all.
    coherence = max(0.0, raw_parity - 2.0 * complementary_pair_bound(z_ps, n, total_z))
    c_err = np.sqrt(max(0.0, 1 - coherence**2) / total_x)

    return population, coherence, p_err, c_err


def estimate_fidelity_oscillation(
    z_counts: dict[str, int],
    osc_counts_list: list[dict[str, int]],
    phases: list[float],
    n: int,
    num_flag_qubits: int,
) -> tuple[float, float, float, float]:
    """Estimate GHZ fidelity using parity oscillation curve fitting.

    Returns: (population, coherence, population_err, coherence_err)
    """
    z_ps = post_select_results(z_counts, num_flag_qubits)
    total_z = sum(z_ps.values())

    if total_z == 0:
        return 0.0, 0.0, 0.0, 0.0

    population = (z_ps.get("0" * n, 0) + z_ps.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    # Compute parities for each phase
    parities = []
    for osc_counts in osc_counts_list:
        ps = post_select_results(osc_counts, num_flag_qubits)
        total = sum(ps.values())
        if total == 0:
            parities.append(0.0)
            continue
        even = sum(c for b, c in ps.items() if b.count("1") % 2 == 0)
        parities.append((2 * even - total) / total)

    # Fit oscillation: amplitude * cos(n * phi + offset)
    def osc_func(phi, amplitude, offset):
        return amplitude * np.cos(n * phi + offset)

    try:
        popt, pcov = curve_fit(osc_func, np.array(phases), np.array(parities), p0=[0.5, 0])
        coherence = abs(popt[0])
        c_err = np.sqrt(pcov[0, 0])
    except Exception as exc:
        logger.warning(
            "Parity oscillation fit failed for GHZ benchmark "
            "(returning coherence=0); root cause: %s",
            exc,
        )
        coherence = 0.0
        c_err = 0.0

    return population, coherence, p_err, c_err


def _cosine_dictionary(phases: np.ndarray, max_freq: int) -> np.ndarray:
    """Sampled cosine/sine dictionary for sparse spectral recovery.

    Columns are [1, cos(φ), sin(φ), cos(2φ), sin(2φ), ..., cos(Kφ), sin(Kφ)]
    evaluated at the sample phases, so a real signal Σ_k a_k cos(kφ + θ_k)
    is a sparse linear combination of columns.
    """
    columns = [np.ones_like(phases)]
    for k in range(1, max_freq + 1):
        columns.append(np.cos(k * phases))
        columns.append(np.sin(k * phases))
    return np.stack(columns, axis=1)


def _lasso_spectrum(
    dictionary: np.ndarray,
    signal: np.ndarray,
    lam_ratio: float = 0.1,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve min_x 0.5·||Ax - y||² + λ·||x||₁ with FISTA.

    λ is set relative to λ_max = max|Aᵀy| (the smallest λ with all-zero
    solution), the standard parameterization for basis-pursuit denoising.
    """
    lam = lam_ratio * float(np.max(np.abs(dictionary.T @ signal), initial=0.0))
    lipschitz = float(np.linalg.norm(dictionary, 2)) ** 2
    x = np.zeros(dictionary.shape[1])
    if lipschitz == 0.0:
        return x
    z = x.copy()
    t = 1.0
    for _ in range(max_iter):
        x_prev = x
        gradient = dictionary.T @ (dictionary @ z - signal)
        w = z - gradient / lipschitz
        x = np.sign(w) * np.maximum(np.abs(w) - lam / lipschitz, 0.0)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        z = x + ((t - 1.0) / t_next) * (x - x_prev)
        t = t_next
        if np.linalg.norm(x - x_prev) <= tol * max(1.0, float(np.linalg.norm(x_prev))):
            break
    return x


def estimate_fidelity_compressed_sensing(
    z_counts: dict[str, int],
    osc_counts_list: list[dict[str, int]],
    phases: list[float],
    n: int,
    num_flag_qubits: int,
) -> tuple[float, float, float, float, int]:
    """Estimate GHZ fidelity via compressed sensing.

    Samples the parity signal at M random phases on [0, 2π) and recovers its
    sparse frequency spectrum by L1-regularized least squares (LASSO) over a
    cosine/sine dictionary, followed by an ordinary least-squares refit on the
    recovered support to remove the L1 shrinkage bias. Random phases provide
    the incoherence needed for sub-Nyquist recovery with M ≈ 5·ln(n) samples.

    The coherence is the amplitude of the n-qubit frequency component. The
    dominant frequency overall is also returned: it equals n only if the full
    GHZ state was prepared, so it acts as a witness of the actual GHZ size
    (a k-qubit entangled core oscillates at frequency k, not n).
    See arXiv:2604.27824.

    Returns: (population, coherence, population_err, coherence_err, recovered_frequency)
    """
    z_ps = post_select_results(z_counts, num_flag_qubits)
    total_z = sum(z_ps.values())

    if total_z == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    population = (z_ps.get("0" * n, 0) + z_ps.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    parities: list[float] = []
    parity_vars: list[float] = []
    for osc_counts in osc_counts_list:
        ps = post_select_results(osc_counts, num_flag_qubits)
        total = sum(ps.values())
        if total == 0:
            parities.append(0.0)
            parity_vars.append(0.0)
            continue
        even = sum(c for b, c in ps.items() if b.count("1") % 2 == 0)
        p_k = (2 * even - total) / total
        parities.append(p_k)
        parity_vars.append((1.0 - p_k**2) / total)

    M = len(parities)
    if M == 0:
        return population, 0.0, p_err, 0.0, 0

    dictionary = _cosine_dictionary(np.asarray(phases), n)
    signal = np.asarray(parities)
    coeffs = _lasso_spectrum(dictionary, signal)

    # Debias: LASSO shrinks the surviving coefficients toward zero, so refit
    # ordinary least squares on the recovered support (the relaxed-lasso
    # estimator), plus frequency n so the coherence is always defined.
    #
    # arXiv:2604.27824 refits only the dominant recovered frequency because it
    # reads only that amplitude. This implementation reads the amplitude at
    # frequency n, which is a different column whenever the state is partially
    # entangled, so truncating the refit to a subset of the recovered support
    # biases what remains: on the cos(10 phi) cos^2(phi) signal, whose exact
    # amplitudes are 1/4, 1/2, 1/4 at frequencies 8, 10, 12, dropping the
    # frequency-8 column pushes the frequency-12 amplitude from 0.25 to 0.32,
    # since the discarded energy has nowhere else to go. Refitting the full
    # support recovers 0.25. Skipped when the system is not overdetermined.
    support = np.flatnonzero(np.abs(coeffs) > 1e-8)
    if 0 < support.size < M:
        coherence_cols = [c for c in (2 * n - 1, 2 * n) if c < dictionary.shape[1]]
        refit_cols = sorted(set(support.tolist()) | set(coherence_cols))
        if len(refit_cols) < M:
            refit, *_ = np.linalg.lstsq(dictionary[:, refit_cols], signal, rcond=None)
            coeffs = np.zeros_like(coeffs)
            coeffs[refit_cols] = refit

    # Fold cos/sin coefficient pairs into per-frequency amplitudes.
    amplitudes = np.empty(n + 1)
    amplitudes[0] = abs(coeffs[0])
    for k in range(1, n + 1):
        amplitudes[k] = float(np.hypot(coeffs[2 * k - 1], coeffs[2 * k]))

    coherence = float(amplitudes[n])
    oscillating = amplitudes[1:]
    recovered_frequency = int(np.argmax(oscillating)) + 1 if np.max(oscillating) > 1e-9 else 0

    # Shot-noise propagation for a least-squares cosine amplitude over M
    # samples; the real and imaginary quadratures each pick up roughly half
    # the total parity variance.
    c_err = (np.sqrt(2.0) / M) * float(np.sqrt(np.sum(parity_vars)))

    return population, coherence, p_err, c_err, recovered_frequency


def bisection_sizes(device_size: int, min_qubits: int) -> list[int]:
    """Halving size grid from the device size down to min_qubits, largest first.

    E.g. device_size=156, min_qubits=5 -> [156, 78, 39, 19, 9]. Every size is
    dispatched in one job, so the cost is logarithmic in the device size while
    still bracketing the largest buildable GHZ state to within a factor of two.
    """
    if device_size < min_qubits:
        raise ValueError(
            f"Device exposes {device_size} qubits, below the search lower bound "
            f"of {min_qubits}"
        )
    sizes: list[int] = []
    n = device_size
    while n >= min_qubits:
        sizes.append(n)
        n //= 2
    # Halving can step over the lower bound (9 -> 4 with min_qubits 5), which
    # would report zero when min_qubits itself would have passed. Costs at most
    # one extra size.
    if sizes[-1] != min_qubits:
        sizes.append(min_qubits)
    return sizes


def estimate_fidelity_for_method(
    method: str,
    counts: list[dict[str, int]],
    phases: list[float],
    n: int,
    num_flag_qubits: int,
) -> tuple[float, float, float, float, int | None]:
    """Estimate (population, coherence, p_err, c_err, recovered_frequency) for one size.

    ``counts`` holds the measurement counts for this size's circuits in
    dispatch order (Z-basis circuit first, then the method's other circuits).
    ``recovered_frequency`` is None for methods other than compressed sensing.
    """
    if method == "dfe":
        pop, coh, p_err, c_err = estimate_fidelity_dfe(counts[0], counts[1], n, num_flag_qubits)
        return pop, coh, p_err, c_err, None
    if method == "compressed_sensing":
        pop, coh, p_err, c_err, recovered = estimate_fidelity_compressed_sensing(
            counts[0], list(counts[1:]), phases, n, num_flag_qubits
        )
        return pop, coh, p_err, c_err, recovered
    pop, coh, p_err, c_err = estimate_fidelity_oscillation(
        counts[0], list(counts[1:]), phases, n, num_flag_qubits
    )
    return pop, coh, p_err, c_err, None


# ---------------------------------------------------------------------------
# Benchmark handler
# ---------------------------------------------------------------------------


class GHZBenchmark(Benchmark):
    def _phase_grid(self, method: str, n: int) -> list[float]:
        """Phase grid for oscillation-style methods.

        parity_oscillation samples the full period [0, 2π) on a uniform grid
        (default 20 points) and relies on a nonlinear fit. compressed_sensing
        samples random phases on [0, 2π), as required for sparse recovery,
        and needs only M ≈ 5·ln(n) samples (the default when `num_phases` is
        not set); pass `seed` for a reproducible draw.
        """
        num_phases = getattr(self.params, "num_phases", None)
        if method == "parity_oscillation":
            # The parity signal oscillates at frequency n, so a uniform grid
            # needs strictly more than 2n points to avoid aliasing: at 20
            # uniform angles cos(100 phi) and cos(80 phi) are pointwise
            # identical, which would read a large state's coherence off the
            # wrong Fourier component.
            min_phases = 2 * n + 1
            if num_phases is None:
                num_phases = max(20, min_phases)
            elif num_phases < min_phases:
                raise ValueError(
                    f"parity_oscillation with num_qubits={n} needs at least "
                    f"{min_phases} uniform phases to avoid aliasing, got {num_phases}; "
                    f"raise num_phases or use compressed_sensing, which needs only "
                    f"about 5*ln(n) random phases"
                )
            return np.linspace(0, 2 * np.pi, num_phases, endpoint=False).tolist()
        if method == "compressed_sensing":
            if num_phases is None:
                num_phases = max(6, math.ceil(5 * math.log(n)))
            rng = np.random.default_rng(getattr(self.params, "seed", None))
            return rng.uniform(0.0, 2 * np.pi, num_phases).tolist()
        return []

    def _resolved_method(self) -> str:
        """Verification method, defaulting per mode.

        Size search defaults to compressed sensing rather than DFE. DFE bounds
        the non-GHZ parity contribution from the observed Z distribution, and
        that bound stops being valid once 2^n outruns the shot count, which the
        largest searched sizes always do.
        """
        method = getattr(self.params, "method", None)
        if method:
            return str(method)
        return "compressed_sensing" if getattr(self.params, "size_search", False) else "dfe"

    def _build_circuits(
        self, device: "QuantumDevice"
    ) -> tuple[
        list[QuantumCircuit], list[float], int, list[int], list[int], list[list[float]], list[int]
    ]:
        """Build circuits for either mode.

        Returns (circuits, phases, num_flags, search_sizes,
        search_circuit_counts, search_phases, search_flag_counts). The search
        lists are empty in fixed-size mode; phases is empty in search mode
        (the per-size grids live in search_phases).

        Flag counts are the number of flags actually selected, which can be
        fewer than requested (or zero at the full device size, where no spare
        qubits remain). Polling must strip exactly the bits that were measured.
        """
        graph = connectivity_graph(device)
        method = self._resolved_method()
        num_flag_qubits = getattr(self.params, "num_flag_qubits", 0)

        if bool(getattr(self.params, "size_search", False)):
            min_qubits = getattr(self.params, "min_qubits", None) or 5
            sizes = bisection_sizes(graph.num_nodes(), min_qubits)
            if method == "dfe":
                shots = getattr(self.params, "shots", 0) or 0
                unsound = [s for s in sizes if not dfe_sampling_is_adequate(s, shots)]
                if unsound:
                    logger.warning(
                        "GHZ size search with method='dfe' at sizes %s: 2^n exceeds "
                        "the %d shots, so the complementary-pair correction cannot be "
                        "estimated and the reported fidelity is not a valid lower "
                        "bound. Use compressed_sensing for these sizes.",
                        unsound,
                        shots,
                    )
            circuits: list[QuantumCircuit] = []
            circuit_counts: list[int] = []
            search_phases: list[list[float]] = []
            flag_counts: list[int] = []
            for size in sizes:
                # Drawn per size so dispatch stores exactly the phases the
                # circuits were built with (random for compressed_sensing).
                phases = self._phase_grid(method, size)
                size_circuits, _, size_flags = build_ghz_circuits(
                    graph=graph,
                    num_qubits=size,
                    method=method,
                    phases=phases or None,
                    num_flag_qubits=num_flag_qubits,
                )
                circuits.extend(size_circuits)
                circuit_counts.append(len(size_circuits))
                search_phases.append(phases)
                flag_counts.append(len(size_flags))
            return circuits, [], 0, sizes, circuit_counts, search_phases, flag_counts

        num_qubits = self.params.num_qubits
        phases = self._phase_grid(method, num_qubits)
        circuits, _, flags = build_ghz_circuits(
            graph=graph,
            num_qubits=num_qubits,
            method=method,
            phases=phases or None,
            num_flag_qubits=num_flag_qubits,
        )
        return circuits, phases, len(flags), [], [], [], []

    def dispatch_handler(self, device: "QuantumDevice") -> GHZData:
        (
            circuits,
            phases,
            num_flags,
            search_sizes,
            search_circuit_counts,
            search_phases,
            search_flag_counts,
        ) = self._build_circuits(device)
        method = self._resolved_method()

        quantum_job = device.run(circuits, shots=self.params.shots)

        # No local transpilation pass, so transpiled counts mirror the input.
        counts = [count_two_qubit_gates(c) for c in circuits]

        return GHZData.from_quantum_job(
            quantum_job,
            num_qubits=search_sizes[0] if search_sizes else self.params.num_qubits,
            method=method,
            phases=phases,
            # Store the flags actually measured, not the number requested.
            num_flag_qubits=num_flags,
            search_sizes=search_sizes,
            search_circuit_counts=search_circuit_counts,
            search_phases=search_phases,
            search_flag_counts=search_flag_counts,
            fidelity_threshold=getattr(self.params, "fidelity_threshold", None) or 0.5,
            confidence_sigma=(
                getattr(self.params, "confidence_sigma", None)
                if getattr(self.params, "confidence_sigma", None) is not None
                else 1.0
            ),
            input_two_qubit_gate_counts=counts,
            transpiled_two_qubit_gate_counts=counts,
        )

    def poll_handler(
        self,
        job_data: GHZData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> GHZResult:
        all_counts = flatten_counts(result_data)
        method = job_data.method
        num_flags = job_data.num_flag_qubits

        if job_data.search_sizes:
            return self._poll_size_search(job_data, all_counts, method, num_flags)

        n = job_data.num_qubits
        pop, coh, p_err, c_err, recovered_frequency = estimate_fidelity_for_method(
            method, all_counts, job_data.phases, n, num_flags
        )
        fidelity = (pop + coh) / 2
        f_err = np.sqrt(p_err**2 + c_err**2) / 2

        return GHZResult(
            population=BenchmarkScore(value=pop, uncertainty=p_err),
            coherence=BenchmarkScore(value=coh, uncertainty=c_err),
            fidelity=BenchmarkScore(value=fidelity, uncertainty=f_err),
            recovered_frequency=recovered_frequency,
        )

    def _poll_size_search(
        self,
        job_data: GHZData,
        all_counts: list[dict[str, int]],
        method: str,
        num_flags: int,
    ) -> GHZResult:
        sizes = job_data.search_sizes
        threshold = job_data.fidelity_threshold
        sigma = job_data.confidence_sigma
        per_size: list[tuple[float, float, float, float, int | None]] = []
        offset = 0
        for i, size in enumerate(sizes):
            count = job_data.search_circuit_counts[i]
            counts_slice = all_counts[offset : offset + count]
            offset += count
            phases = job_data.search_phases[i] if job_data.search_phases else []
            # Flags are selected per size, so strip exactly the bits measured
            # for this size rather than a single global count.
            size_flags = (
                job_data.search_flag_counts[i] if job_data.search_flag_counts else num_flags
            )
            per_size.append(
                estimate_fidelity_for_method(method, counts_slice, phases, size, size_flags)
            )

        fidelities = [(pop + coh) / 2 for pop, coh, _, _, _ in per_size]
        uncertainties = [
            float(np.sqrt(p_err**2 + c_err**2) / 2) for _, _, p_err, c_err, _ in per_size
        ]

        # Sizes are searched largest-first; the first one clearing the
        # threshold is the largest passing size. Require the threshold to be
        # cleared by sigma standard deviations so a size sitting right at the
        # boundary does not flip the discrete score between runs.
        passing_index: int | None = None
        for i, fid in enumerate(fidelities):
            if fid - sigma * uncertainties[i] > threshold:
                passing_index = i
                break

        device_size = sizes[0]
        largest_passing = sizes[passing_index] if passing_index is not None else 0
        # Headline population/coherence/fidelity describe the largest passing
        # size; when nothing passes, the smallest size attempted (the one most
        # likely to have succeeded).
        headline = passing_index if passing_index is not None else len(sizes) - 1
        pop, coh, p_err, c_err, recovered_frequency = per_size[headline]

        return GHZResult(
            population=BenchmarkScore(value=pop, uncertainty=p_err),
            coherence=BenchmarkScore(value=coh, uncertainty=c_err),
            fidelity=BenchmarkScore(
                value=fidelities[headline], uncertainty=uncertainties[headline]
            ),
            recovered_frequency=recovered_frequency,
            largest_passing_size=BenchmarkScore(value=largest_passing),
            device_fraction=BenchmarkScore(value=largest_passing / device_size),
            search_sizes=list(sizes),
            search_fidelities=fidelities,
            search_uncertainties=uncertainties,
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        circuits, *_ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
