"""GHZ state preparation and verification benchmark.

Summary:
    Constructs a GHZ state on the device using a binary dissemination tree over
    the device's connectivity graph (logarithmic depth where connectivity
    allows), then estimates fidelity via one of four verification methods:
    direct fidelity estimation (DFE, uniform sampling of the GHZ stabilizer
    group), the two-setting Z/X lower bound, parity oscillation curve fitting,
    or compressed sensing (parity samples at random phases, sparse spectrum
    recovery by L1 minimization).

    **Experimental:** available for standalone evaluation. The methodology and
    defaults may change before inclusion in the benchmark suite.

    Two modes:

    - Fixed size (`num_qubits`): verify a GHZ state of one requested size.
    - Size search (`size_search: true`): build GHZ states at halving sizes
      from the device size down to `min_qubits` (e.g. 156, 78, 39, 19, 9, 5),
      all dispatched in a single job, and report the largest size whose
      fidelity clears `fidelity_threshold` by `confidence_sigma` standard
      deviations. Defaults to compressed sensing, which resolves the
      frequency spectrum using logarithmically many phase samples.

Result interpretation:
    - population: probability of measuring all-zero or all-one bitstrings (Z basis).
    - coherence: off-diagonal element. DFE averages uniformly sampled
      off-diagonal stabilizers X^(x)N·Z_s, an unbiased estimate of
      2·Re(rho[0...0, 1...1]) with pure finite-sampling error (the diagonal
      half of the stabilizer group averages exactly to the population, so the
      Z circuit covers it with no sampling). two_setting_bound takes the
      global X parity and subtracts the population outside the GHZ subspace,
      bounding the contribution of non-GHZ complementary pairs: X^(x)N connects every
      bitstring to its complement and not only 0...0 to 1...1; without that
      subtraction a product state such as |+>^(x)N reports coherence 1. The
      oscillation methods instead isolate the N-qubit Fourier component, by
      curve fit or by LASSO over random-phase parity samples.
    - fidelity: (population + coherence) / 2. DFE targets GHZ+; oscillation
      methods maximize over the relative phase, and two_setting_bound bounds
      that phase-optimized fidelity. These targets differ for coherent phase errors.
      Compare runs using the same method.
    - recovered_frequency (compressed sensing only): dominant frequency of the
      parity signal. This is a diagnostic of the spectrum, not by itself an
      entanglement certificate; use fidelity to assess the prepared state.
    - phase_offset (compressed sensing only): phase θ of the off-diagonal
      element, i.e. the state (|0...0> + e^{iθ}|1...1>)/√2. Zero for an ideal
      GHZ; a nonzero value exposes coherent errors (residual Z rotations,
      frame misalignment) that leave the coherence magnitude unchanged. None
      when the coherence is at the noise floor, where the angle is undefined.
    - largest_passing_size / device_fraction (size search only): the largest
      tested size that passes the fidelity margin, and that size divided
      by the device size. Per-size fidelities are reported in search_sizes /
      search_fidelities, with uncertainties in search_uncertainties, so no
      resolution is lost to the threshold.
    - score: fidelity in fixed-size mode; largest_passing_size (a qubit count)
      in size-search mode. Scores from these modes are not directly comparable.

Size search and uncertainty:
    A size passes when
    `fidelity - confidence_sigma * uncertainty > fidelity_threshold`.
    This margin uses estimated standard errors, not a rigorous confidence
    guarantee. Size search reports the largest **tested** passing size.

    Compressed sensing assumes a sparse spectrum. Its uncertainty propagates
    shot noise through the recovered model. If the recovered spectrum needs
    more phase samples, increase `num_phases` or use uniform parity oscillation.

References:
    - Cruz et al., "Efficient quantum algorithms for GHZ and W states, and
      implementation on the IBM quantum computer", Adv. Quantum Technol. 2,
      1900015 (2019). [arXiv:1807.05572](https://arxiv.org/abs/1807.05572).
    - Flammia & Liu, "Direct Fidelity Estimation from Few Pauli Measurements",
      Phys. Rev. Lett. 106, 230501 (2011).
      [arXiv:1104.4695](https://arxiv.org/abs/1104.4695).
    - Labib et al., "Compressed Sensing for Efficient Fidelity Estimation of
      GHZ States" (2026). [arXiv:2604.27824](https://arxiv.org/abs/2604.27824).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field
from qbraid.runtime import QiskitBackend
from qiskit import QuantumCircuit, transpile
from typing import TYPE_CHECKING

from metriq_gym.benchmarks.benchmark import (
    Benchmark,
    BenchmarkData,
    BenchmarkResult,
    BenchmarkScore,
)
from metriq_gym.benchmarks.ghz.analysis import (
    _cosine_dictionary as _cosine_dictionary,
    _fidelity_uncertainty as _fidelity_uncertainty,
    _lasso_spectrum as _lasso_spectrum,
    estimate_fidelity_compressed_sensing as estimate_fidelity_compressed_sensing,
    estimate_fidelity_dfe as estimate_fidelity_dfe,
    estimate_fidelity_for_method as estimate_fidelity_for_method,
    estimate_fidelity_oscillation as estimate_fidelity_oscillation,
    estimate_fidelity_two_setting_bound as estimate_fidelity_two_setting_bound,
)
from metriq_gym.benchmarks.ghz.circuits import (
    _bfs_edges as _bfs_edges,
    _dissemination_rounds as _dissemination_rounds,
    build_ghz_circuits as build_ghz_circuits,
    sample_ghz_stabilizers as sample_ghz_stabilizers,
)
from metriq_gym.helpers.task_helpers import flatten_counts
from metriq_gym.local.device import LocalAerDevice
from metriq_gym.qplatform.device import connectivity_graph
from metriq_gym.resource_estimation import CircuitBatch, count_two_qubit_gates

if TYPE_CHECKING:
    from qbraid import GateModelResultData, QuantumDevice, QuantumJob


# ---------------------------------------------------------------------------
# Data / Result classes
# ---------------------------------------------------------------------------


@dataclass
class GHZData(BenchmarkData):
    num_qubits: int = 0
    method: str = "dfe"
    phases: list[float] = field(default_factory=list)
    # DFE: sampled off-diagonal stabilizers, each the list of data-qubit
    # positions measured in the Y basis (the support of the Z string s in
    # X^(x)N·Z_s). Polling needs them for the (-1)^{|s|/2} signs.
    stabilizers: list[list[int]] = field(default_factory=list)
    # Size-search mode: the halving size grid, how many circuits belong to
    # each size (in dispatch order), the per-size phase grids, and the
    # fidelity threshold a size must clear. Empty lists mean fixed-size mode.
    search_sizes: list[int] = field(default_factory=list)
    search_circuit_counts: list[int] = field(default_factory=list)
    confidence_sigma: float = 1.0
    search_phases: list[list[float]] = field(default_factory=list)
    search_stabilizers: list[list[list[int]]] = field(default_factory=list)
    fidelity_threshold: float = 0.5


class GHZResult(BenchmarkResult):
    population: BenchmarkScore = Field(description="Z-basis population of |0...0> + |1...1>")
    coherence: BenchmarkScore = Field(
        description="Signed GHZ+ coherence (DFE), magnitude (oscillation), or magnitude lower bound"
    )
    fidelity: BenchmarkScore = Field(
        description="GHZ fidelity estimate or two-setting lower bound: (population + coherence) / 2"
    )
    recovered_frequency: int | None = Field(
        default=None,
        description=(
            "Dominant parity-oscillation frequency recovered by compressed sensing; "
            "diagnoses the signal spectrum but does not alone certify entanglement. "
            "None for other methods."
        ),
    )
    phase_offset: float | None = Field(
        default=None,
        description=(
            "Phase θ of the GHZ off-diagonal element recovered by compressed "
            "sensing, i.e. the prepared state is (|0...0> + e^{iθ}|1...1>)/√2. "
            "Zero for an ideal GHZ; nonzero values indicate coherent errors. "
            "None for other methods or when the coherence is at the noise floor."
        ),
    )
    phase_offset_err: float | None = Field(
        default=None,
        description="Propagated shot uncertainty on phase_offset; None when phase_offset is None.",
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


def bisection_sizes(device_size: int, min_qubits: int) -> list[int]:
    """Halving size grid from the device size down to min_qubits, largest first.

    E.g. device_size=156, min_qubits=5 -> [156, 78, 39, 19, 9, 5]. Every size is
    dispatched in one job, so the cost is logarithmic in the device size while
    still bracketing the largest buildable GHZ state to within a factor of two.
    """
    if device_size < min_qubits:
        raise ValueError(
            f"Device exposes {device_size} qubits, below the search lower bound of {min_qubits}"
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


# ---------------------------------------------------------------------------
# Benchmark handler
# ---------------------------------------------------------------------------


class GHZBenchmark(Benchmark):
    def _phase_grid(self, method: str, n: int) -> list[float]:
        """Phase grid for oscillation-style methods.

        parity_oscillation samples the full period [0, 2π) on a uniform grid
        (default max(20, 2n+1) points) and fits both quadratures. compressed_sensing
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

    def _stabilizer_sample(self, method: str, n: int) -> list[list[int]]:
        """Sampled off-diagonal stabilizers for dfe; empty for other methods.

        ell = O(1/eps^2) independent of n (Flammia-Liu); the default of 10
        keeps the circuit count comparable to compressed sensing. Pass `seed`
        for a reproducible draw.
        """
        if method != "dfe":
            return []
        num_stabilizers = getattr(self.params, "num_stabilizers", None) or 10
        return sample_ghz_stabilizers(n, num_stabilizers, getattr(self.params, "seed", None))

    def _resolved_method(self) -> str:
        """Verification method, defaulting per mode.

        Size search defaults to compressed sensing for an estimate of the
        phase-optimized fidelity; the inexpensive two-setting method gives
        only a lower bound.
        """
        method = getattr(self.params, "method", None)
        if method:
            return str(method)
        return "compressed_sensing" if getattr(self.params, "size_search", False) else "dfe"

    def _build_circuits(
        self, device: "QuantumDevice"
    ) -> tuple[
        list[QuantumCircuit],
        list[float],
        list[list[int]],
        list[int],
        list[int],
        list[list[float]],
        list[list[list[int]]],
    ]:
        """Build circuits for either mode.

        Returns (circuits, phases, stabilizers, search_sizes,
        search_circuit_counts, search_phases, search_stabilizers). The search
        lists are empty in fixed-size mode; phases/stabilizers are empty in
        search mode (the per-size draws live in search_phases/
        search_stabilizers).
        """
        graph = connectivity_graph(device)
        method = self._resolved_method()

        if bool(getattr(self.params, "size_search", False)):
            min_qubits = getattr(self.params, "min_qubits", None) or 5
            sizes = bisection_sizes(graph.num_nodes(), min_qubits)
            circuits: list[QuantumCircuit] = []
            circuit_counts: list[int] = []
            search_phases: list[list[float]] = []
            search_stabilizers: list[list[list[int]]] = []
            for size in sizes:
                # Drawn per size so dispatch stores exactly the phases and
                # stabilizers the circuits were built with.
                phases = self._phase_grid(method, size)
                stabilizers = self._stabilizer_sample(method, size)
                size_circuits, _ = build_ghz_circuits(
                    graph=graph,
                    num_qubits=size,
                    method=method,
                    phases=phases or None,
                    stabilizers=stabilizers or None,
                )
                circuits.extend(size_circuits)
                circuit_counts.append(len(size_circuits))
                search_phases.append(phases)
                search_stabilizers.append(stabilizers)
            return (
                circuits,
                [],
                [],
                sizes,
                circuit_counts,
                search_phases,
                search_stabilizers,
            )

        num_qubits = self.params.num_qubits
        phases = self._phase_grid(method, num_qubits)
        stabilizers = self._stabilizer_sample(method, num_qubits)
        circuits, _ = build_ghz_circuits(
            graph=graph,
            num_qubits=num_qubits,
            method=method,
            phases=phases or None,
            stabilizers=stabilizers or None,
        )
        return circuits, phases, stabilizers, [], [], [], []

    def dispatch_handler(self, device: "QuantumDevice") -> GHZData:
        (
            circuits,
            phases,
            stabilizers,
            search_sizes,
            search_circuit_counts,
            search_phases,
            search_stabilizers,
        ) = self._build_circuits(device)
        method = self._resolved_method()

        counts = [count_two_qubit_gates(c) for c in circuits]
        transpiled_counts = counts
        if isinstance(device, (QiskitBackend, LocalAerDevice)):
            # These circuits already use physical device indices. qBraid's
            # default transform selects a fresh layout for each measurement
            # setting, which would combine observations of different physical
            # GHZ preparations. Translate gates without changing that layout
            # or optimizing the common preparation differently per setting.
            circuits = [
                transpile(
                    circuit,
                    backend=device._backend,
                    initial_layout=list(range(circuit.num_qubits)),
                    optimization_level=0,
                )
                for circuit in circuits
            ]
            transpiled_counts = [count_two_qubit_gates(c) for c in circuits]
            original_transform = device._options.get("transform")
            device.set_options(transform=False)
            try:
                quantum_job = device.run(circuits, shots=self.params.shots)
            finally:
                device.set_options(transform=original_transform)
        else:
            quantum_job = device.run(circuits, shots=self.params.shots)

        return GHZData.from_quantum_job(
            quantum_job,
            num_qubits=search_sizes[0] if search_sizes else self.params.num_qubits,
            method=method,
            phases=phases,
            stabilizers=stabilizers,
            search_sizes=search_sizes,
            search_circuit_counts=search_circuit_counts,
            search_phases=search_phases,
            search_stabilizers=search_stabilizers,
            fidelity_threshold=getattr(self.params, "fidelity_threshold", None) or 0.5,
            confidence_sigma=(
                getattr(self.params, "confidence_sigma", None)
                if getattr(self.params, "confidence_sigma", None) is not None
                else 1.0
            ),
            input_two_qubit_gate_counts=counts,
            transpiled_two_qubit_gate_counts=transpiled_counts,
        )

    def poll_handler(
        self,
        job_data: GHZData,
        result_data: list["GateModelResultData"],
        quantum_jobs: list["QuantumJob"],
    ) -> GHZResult:
        all_counts = flatten_counts(result_data)
        method = job_data.method

        if job_data.search_sizes:
            return self._poll_size_search(job_data, all_counts, method)

        n = job_data.num_qubits
        pop, coh, p_err, c_err, recovered_frequency, phase, phase_err = (
            estimate_fidelity_for_method(
                method, all_counts, job_data.phases, n, job_data.stabilizers
            )
        )
        fidelity = (pop + coh) / 2
        f_err = _fidelity_uncertainty(method, p_err, c_err)

        return GHZResult(
            population=BenchmarkScore(value=pop, uncertainty=p_err),
            coherence=BenchmarkScore(value=coh, uncertainty=c_err),
            fidelity=BenchmarkScore(value=fidelity, uncertainty=f_err),
            recovered_frequency=recovered_frequency,
            phase_offset=phase,
            phase_offset_err=phase_err,
        )

    def _poll_size_search(
        self,
        job_data: GHZData,
        all_counts: list[dict[str, int]],
        method: str,
    ) -> GHZResult:
        sizes = job_data.search_sizes
        threshold = job_data.fidelity_threshold
        sigma = job_data.confidence_sigma
        per_size: list[
            tuple[float, float, float, float, int | None, float | None, float | None]
        ] = []
        offset = 0
        for i, size in enumerate(sizes):
            count = job_data.search_circuit_counts[i]
            counts_slice = all_counts[offset : offset + count]
            offset += count
            phases = job_data.search_phases[i] if job_data.search_phases else []
            stabilizers = job_data.search_stabilizers[i] if job_data.search_stabilizers else []
            per_size.append(
                estimate_fidelity_for_method(method, counts_slice, phases, size, stabilizers)
            )

        fidelities = [(pop + coh) / 2 for pop, coh, *_ in per_size]
        uncertainties = [
            _fidelity_uncertainty(method, p_err, c_err) for _, _, p_err, c_err, *_ in per_size
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
        pop, coh, p_err, c_err, recovered_frequency, phase, phase_err = per_size[headline]

        return GHZResult(
            population=BenchmarkScore(value=pop, uncertainty=p_err),
            coherence=BenchmarkScore(value=coh, uncertainty=c_err),
            fidelity=BenchmarkScore(
                value=fidelities[headline], uncertainty=uncertainties[headline]
            ),
            recovered_frequency=recovered_frequency,
            phase_offset=phase,
            phase_offset_err=phase_err,
            largest_passing_size=BenchmarkScore(value=largest_passing),
            device_fraction=BenchmarkScore(value=largest_passing / device_size),
            search_sizes=list(sizes),
            search_fidelities=fidelities,
            search_uncertainties=uncertainties,
        )

    def estimate_resources_handler(self, device: "QuantumDevice") -> list[CircuitBatch]:
        circuits, *_ = self._build_circuits(device)
        return [CircuitBatch(circuits=circuits, shots=self.params.shots)]
