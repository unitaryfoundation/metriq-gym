"""GHZ state preparation and verification benchmark.

Summary:
    Constructs a GHZ state on the device using a BFS spanning tree of the
    device's connectivity graph, then estimates fidelity via one of three
    verification methods: direct fidelity estimation (DFE), parity
    oscillation curve fitting, or compressed sensing (parity samples at
    random phases, sparse spectrum recovery by L1 minimization).

Result interpretation:
    - population: probability of measuring all-zero or all-one bitstrings (Z basis).
    - coherence: off-diagonal element magnitude, measured via X-basis parity
      (DFE, fidelity lower bound), fitted amplitude of the parity oscillation
      curve, or amplitude of the N-qubit frequency component recovered by
      LASSO from random-phase parity samples (compressed sensing).
    - fidelity: (population + coherence) / 2.
    - recovered_frequency (compressed sensing only): dominant frequency of the
      parity signal, which equals N only if the full N-qubit GHZ state was
      actually prepared; a partially entangled state shows a lower frequency.

References:
    - Moses et al., "A Race-Track Trapped-Ion Quantum Processor",
      Phys. Rev. X 13, 041052 (2023). [arXiv:2305.03828]
    - Russo et al., "Compressed Sensing Verification of Large Entangled States",
      arXiv:2409.15302 (2024).
"""

import logging
import math
from collections import deque
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

    def compute_score(self) -> BenchmarkScore:
        return self.fidelity


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------


def _bfs_edges(graph, root: int, num_qubits: int) -> list[tuple[int, int]]:
    """BFS spanning tree edges from root, limited to num_qubits nodes."""
    visited = {root}
    queue = deque([root])
    edges: list[tuple[int, int]] = []

    while queue and len(visited) < num_qubits:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor not in visited and len(visited) < num_qubits:
                visited.add(neighbor)
                queue.append(neighbor)
                edges.append((node, neighbor))

    return edges


def _select_flag_qubits(
    graph, data_qubits: set[int], num_flags: int
) -> list[int]:
    """Select flag qubits: nodes adjacent to data qubits but not in the data set."""
    candidates = set()
    for dq in data_qubits:
        for neighbor in graph.neighbors(dq):
            if neighbor not in data_qubits:
                candidates.add(neighbor)
    return sorted(candidates)[:num_flags]


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

    # Select flag qubits from neighbors not in data set
    flag_qubits = _select_flag_qubits(graph, set(data_qubits), num_flag_qubits)

    n_total = total_device_qubits
    n_clbits = len(data_qubits) + len(flag_qubits)

    def _make_ghz_circuit() -> QuantumCircuit:
        """Create base GHZ state preparation circuit."""
        qc = QuantumCircuit(n_total, n_clbits)
        # Hadamard on root
        qc.h(data_qubits[0])
        # CNOT chain from BFS edges
        for ctrl, targ in bfs:
            qc.cx(ctrl, targ)
        qc.barrier()
        # ZZ flag operations
        for flag in flag_qubits:
            for neighbor in graph.neighbors(flag):
                if neighbor in set(data_qubits):
                    qc.cx(neighbor, flag)
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


def estimate_fidelity_dfe(
    z_counts: dict[str, int],
    x_counts: dict[str, int],
    n: int,
    num_flag_qubits: int,
) -> tuple[float, float, float, float]:
    """Estimate GHZ fidelity using direct fidelity estimation.

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

    # Use the magnitude of the X-basis parity. The signed value distinguishes
    # GHZ+ from GHZ-, but for a fidelity *lower bound* of the form
    # (population + coherence) / 2 we want the off-diagonal magnitude so that
    # GHZ-like states with a relative phase are not penalized.
    even_x = sum(c for b, c in x_ps.items() if b.count("1") % 2 == 0)
    coherence = abs((2 * even_x - total_x) / total_x)
    c_err = np.sqrt((1 - coherence**2) / total_x)

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
    See Russo et al., arXiv:2409.15302.

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
    # ordinary least squares on the recovered support (skipped when the
    # support is not overdetermined, where the refit would just interpolate).
    support = np.flatnonzero(np.abs(coeffs) > 1e-8)
    if 0 < support.size < M:
        refit, *_ = np.linalg.lstsq(dictionary[:, support], signal, rcond=None)
        coeffs = np.zeros_like(coeffs)
        coeffs[support] = refit

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
            return np.linspace(0, 2 * np.pi, num_phases or 20, endpoint=False).tolist()
        if method == "compressed_sensing":
            if num_phases is None:
                num_phases = max(6, math.ceil(5 * math.log(n)))
            rng = np.random.default_rng(getattr(self.params, "seed", None))
            return rng.uniform(0.0, 2 * np.pi, num_phases).tolist()
        return []

    def _build_circuits(
        self, device: "QuantumDevice"
    ) -> tuple[list[QuantumCircuit], list[int], list[int], list[float]]:
        graph = connectivity_graph(device)
        num_qubits = self.params.num_qubits
        method = getattr(self.params, "method", "dfe")
        num_flag_qubits = getattr(self.params, "num_flag_qubits", 0)

        # Drawn once and returned so that dispatch stores exactly the phases
        # the circuits were built with (they are random for compressed_sensing).
        phases = self._phase_grid(method, num_qubits)

        circuits, data_qubits, flag_qubits = build_ghz_circuits(
            graph=graph,
            num_qubits=num_qubits,
            method=method,
            phases=phases or None,
            num_flag_qubits=num_flag_qubits,
        )
        return circuits, data_qubits, flag_qubits, phases

    def dispatch_handler(self, device: "QuantumDevice") -> GHZData:
        circuits, _data_qubits, _flag_qubits, phases = self._build_circuits(device)
        method = getattr(self.params, "method", "dfe")
        num_flag_qubits = getattr(self.params, "num_flag_qubits", 0)

        quantum_job = device.run(circuits, shots=self.params.shots)

        # No local transpilation pass, so transpiled counts mirror the input.
        counts = [count_two_qubit_gates(c) for c in circuits]

        return GHZData.from_quantum_job(
            quantum_job,
            num_qubits=self.params.num_qubits,
            method=method,
            phases=phases,
            num_flag_qubits=num_flag_qubits,
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
        n = job_data.num_qubits
        method = job_data.method
        num_flags = job_data.num_flag_qubits

        recovered_frequency: int | None = None
        if method == "dfe":
            z_counts, x_counts = all_counts[0], all_counts[1]
            pop, coh, p_err, c_err = estimate_fidelity_dfe(z_counts, x_counts, n, num_flags)
        elif method == "compressed_sensing":
            z_counts = all_counts[0]
            osc_counts = list(all_counts[1:])
            pop, coh, p_err, c_err, recovered_frequency = estimate_fidelity_compressed_sensing(
                z_counts, osc_counts, job_data.phases, n, num_flags
            )
        else:
            z_counts = all_counts[0]
            osc_counts = list(all_counts[1:])
            pop, coh, p_err, c_err = estimate_fidelity_oscillation(
                z_counts, osc_counts, job_data.phases, n, num_flags
            )

        fidelity = (pop + coh) / 2
        f_err = np.sqrt(p_err**2 + c_err**2) / 2

        return GHZResult(
            population=BenchmarkScore(value=pop, uncertainty=p_err),
            coherence=BenchmarkScore(value=coh, uncertainty=c_err),
            fidelity=BenchmarkScore(value=fidelity, uncertainty=f_err),
            recovered_frequency=recovered_frequency,
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        circuits, _, _, _ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
