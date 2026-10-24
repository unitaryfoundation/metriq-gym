"""GHZ fidelity estimates and their uncertainties."""

import math

import numpy as np


def estimate_fidelity_two_setting_bound(
    z_counts: dict[str, int],
    x_counts: dict[str, int],
    n: int,
) -> tuple[float, float, float, float]:
    """Estimate a GHZ fidelity lower bound from two settings (Z and X).

    Not direct fidelity estimation: X^(x)N is a single element of the
    stabilizer group, so this yields a conservative bound rather than an
    unbiased estimate — see :func:`estimate_fidelity_dfe` for the real thing.
    The non-GHZ complementary pairs contribute at most the population outside
    the GHZ subspace: 2*sqrt(p_x*p_complement) <= p_x+p_complement. Subtracting
    1-population from |X parity| therefore bounds the coherence magnitude
    without estimating exponentially many individual outcome probabilities.
    Finite-shot uncertainty still applies to both measured expectations.

    Returns: (population, coherence, population_err, coherence_err)
    """
    total_z = sum(z_counts.values())
    total_x = sum(x_counts.values())

    if total_z == 0 or total_x == 0:
        return 0.0, 0.0, 0.0, 0.0

    population = (z_counts.get("0" * n, 0) + z_counts.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    # Magnitude of the X-basis parity. The signed value distinguishes GHZ+ from
    # GHZ-, but for a fidelity lower bound of the form
    # (population + coherence) / 2 we want the off-diagonal magnitude so that
    # GHZ-like states with a relative phase are not penalized.
    even_x = sum(c for b, c in x_counts.items() if b.count("1") % 2 == 0)
    raw_parity = abs((2 * even_x - total_x) / total_x)

    coherence = max(0.0, raw_parity - (1.0 - population))
    c_err = np.sqrt(p_err**2 + max(0.0, 1 - raw_parity**2) / total_x)

    return population, coherence, p_err, c_err


def _fidelity_uncertainty(method: str, p_err: float, c_err: float) -> float:
    """Propagate uncertainty, accounting for shared Z counts in the bound."""
    variance = p_err**2 + c_err**2
    if method == "two_setting_bound":
        # C = |X| + P - 1 shares P with F = (P+C)/2, so Cov(P,C)=Var(P).
        # Keep this conservative error even when C is clamped to zero.
        variance += 2 * p_err**2
    return math.sqrt(variance) / 2


def estimate_fidelity_dfe(
    z_counts: dict[str, int],
    stab_counts_list: list[dict[str, int]],
    stabilizers: list[list[int]],
    n: int,
) -> tuple[float, float, float, float]:
    """Estimate GHZ fidelity by direct fidelity estimation (Flammia-Liu).

    For a stabilizer state, F = 2^-N · sum over the stabilizer group of <S>,
    estimated by uniform sampling. The GHZ group structure makes both halves
    cheap:

    - Diagonal half (even-weight Z strings): the character sum over the
      even-weight subgroup vanishes except on 0^N and 1^N, so its exact
      average IS the population — the single Z circuit covers it with no
      sampling at all.
    - Off-diagonal half (X^(x)N·Z_s): each sampled element is measured by one
      X/Y-basis circuit; averaging the signed parities (-1)^(|s|/2)·parity
      gives an unbiased estimate of the true coherence 2·Re(rho[0^N, 1^N]).

    Unlike the two-setting bound there is no Z-distribution correction and no
    shot-count validity limit; the error is pure finite sampling. The
    coherence is signed: a GHZ state with relative phase pi honestly scores
    fidelity 0 against GHZ+, where the two-setting bound's |parity| would
    report 1.

    Returns: (population, coherence, population_err, coherence_err)
    """
    total_z = sum(z_counts.values())
    if total_z == 0 or not stab_counts_list:
        return 0.0, 0.0, 0.0, 0.0

    population = (z_counts.get("0" * n, 0) + z_counts.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    values: list[float] = []
    shot_vars: list[float] = []
    for y_positions, counts in zip(stabilizers, stab_counts_list):
        total = sum(counts.values())
        if total == 0:
            continue
        even = sum(c for b, c in counts.items() if b.count("1") % 2 == 0)
        parity = (2 * even - total) / total
        # X·Z = -iY on each qubit in s, so X^(x)N·Z_s = (-1)^(|s|/2) × the
        # measured X/Y tensor; |s| is even by construction.
        sign = 1.0 if len(y_positions) % 4 == 0 else -1.0
        values.append(sign * parity)
        shot_vars.append((1 - parity**2) / total)

    if not values:
        return population, 0.0, float(p_err), 0.0

    ell = len(values)
    coherence = float(np.mean(values))
    # The spread across sampled stabilizers captures both the stabilizer
    # sampling variance and shot noise, but underestimates when all sampled
    # values happen to coincide, so the propagated shot noise is kept as a
    # floor. ell == 1 has no spread and falls back to shot noise alone.
    shot_var = float(np.sum(shot_vars)) / ell**2
    sample_var = float(np.var(values, ddof=1)) / ell if ell > 1 else 0.0
    c_err = float(np.sqrt(max(sample_var, shot_var)))

    return population, coherence, float(p_err), c_err


def estimate_fidelity_oscillation(
    z_counts: dict[str, int],
    osc_counts_list: list[dict[str, int]],
    phases: list[float],
    n: int,
) -> tuple[float, float, float, float]:
    """Estimate GHZ fidelity from the sine/cosine quadratures at frequency n.

    The uniform phase grid makes these columns orthogonal to other Fourier
    components. A linear fit stays identifiable even at zero phase offset,
    where a nonlinear amplitude/phase fit can return infinite covariance.

    Returns: (population, coherence, population_err, coherence_err)
    """
    total_z = sum(z_counts.values())

    if total_z == 0:
        return 0.0, 0.0, 0.0, 0.0

    population = (z_counts.get("0" * n, 0) + z_counts.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    # Compute parities for each phase
    parities = []
    parity_vars = []
    for osc_counts in osc_counts_list:
        total = sum(osc_counts.values())
        if total == 0:
            raise ValueError("GHZ parity oscillation requires counts for every phase")
        even = sum(c for b, c in osc_counts.items() if b.count("1") % 2 == 0)
        parity = (2 * even - total) / total
        parities.append(parity)
        parity_vars.append((1 - parity**2) / total)

    angles: np.ndarray = n * np.asarray(phases)
    design = np.column_stack((np.cos(angles), np.sin(angles)))
    quadratures, _, rank, _ = np.linalg.lstsq(design, np.asarray(parities), rcond=None)
    if rank < 2:
        raise ValueError("GHZ parity phases cannot resolve both coherence quadratures")
    coherence = float(np.linalg.norm(quadratures))
    inverse = np.linalg.pinv(design)
    covariance = (inverse * np.asarray(parity_vars)) @ inverse.T
    if coherence > 1e-12:
        direction = quadratures / coherence
        variance = float(direction @ covariance @ direction)
    else:
        # At zero amplitude its gradient is undefined; retain the largest
        # quadrature variance instead of reporting zero uncertainty.
        variance = float(np.linalg.eigvalsh(covariance)[-1])
    c_err = math.sqrt(max(0.0, variance))

    return population, coherence, float(p_err), c_err


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
    lam_floor: float = 0.0,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve min_x 0.5·||Ax - y||² + λ·||x||₁ with FISTA.

    λ is set relative to λ_max = max|Aᵀy| (the smallest λ with all-zero
    solution), the standard parameterization for basis-pursuit denoising.
    """
    lam = max(lam_floor, lam_ratio * float(np.max(np.abs(dictionary.T @ signal), initial=0.0)))
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
) -> tuple[float, float, float, float, int, float | None, float | None]:
    """Estimate GHZ fidelity via compressed sensing.

    Samples the parity signal at M random phases on [0, 2π) and recovers its
    sparse frequency spectrum by L1-regularized least squares (LASSO) over a
    cosine/sine dictionary, followed by an ordinary least-squares refit on both
    quadratures of each recovered frequency to remove the L1 shrinkage bias.
    The reported uncertainty propagates shot noise conditional on this sparse
    model; it does not bound errors from unidentified frequency components.
    Random phases provide the incoherence needed for sub-Nyquist recovery
    with M ≈ 5·ln(n) samples.

    The coherence is the amplitude of the n-qubit frequency component. The
    dominant frequency overall is also returned as a diagnostic. A partially
    entangled state can have several spectral peaks, including frequency n;
    the dominant frequency alone does not certify entanglement.
    See arXiv:2604.27824.

    The frequency-n quadrature coefficients also encode the phase θ of the
    off-diagonal element — the prepared state is (|0...0> + e^{iθ}|1...1>)/√2 —
    which is recovered at no extra cost. The measured per-qubit operator is
    cos(φ)X − sin(φ)Y (Rz(φ) then H then a Z measurement), so the coherence
    signal is A·cos(nφ + θ) and θ = atan2(−b_n, a_n). The angle is undefined
    when the amplitude sits at the noise floor, so it is None there.

    Returns: (population, coherence, population_err, coherence_err,
    recovered_frequency, phase_offset, phase_offset_err)
    """
    total_z = sum(z_counts.values())

    if total_z == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, None, None

    population = (z_counts.get("0" * n, 0) + z_counts.get("1" * n, 0)) / total_z
    p_err = np.sqrt(population * (1 - population) / total_z)

    parities: list[float] = []
    parity_vars: list[float] = []
    for osc_counts in osc_counts_list:
        total = sum(osc_counts.values())
        if total == 0:
            parities.append(0.0)
            parity_vars.append(0.0)
            continue
        even = sum(c for b, c in osc_counts.items() if b.count("1") % 2 == 0)
        p_k = (2 * even - total) / total
        parities.append(p_k)
        # A one-count continuity floor keeps saturated observations from
        # claiming exact parity, especially when only one shot was taken.
        parity_vars.append(max((1.0 - p_k**2) / total, 1.0 / total**2))

    M = len(parities)
    if M == 0:
        return population, 0.0, p_err, 0.0, 0, None, None

    dictionary = _cosine_dictionary(np.asarray(phases), n)
    signal = np.asarray(parities)
    # A purely relative penalty promotes many noise peaks when the physical
    # signal is weak, exhausting the phase samples during the paired refit.
    # Set a noise floor from the largest propagated column standard error,
    # with the usual sqrt(2 log(number of columns)) multiple-comparison scale.
    column_vars = (dictionary**2).T @ np.asarray(parity_vars)
    lam_floor = math.sqrt(2.0 * math.log(dictionary.shape[1]) * float(np.max(column_vars)))
    coeffs = _lasso_spectrum(dictionary, signal, lam_floor=lam_floor)

    # Debias all recovered frequencies, including BOTH quadratures even when
    # LASSO selected only one. Otherwise a small omitted sine coefficient at a
    # lower frequency leaks into the n-qubit amplitude on a random phase grid
    # and can falsely certify a biseparable state. Always include frequency n.
    support = np.flatnonzero(np.abs(coeffs) > 1e-8)
    frequencies = {(int(col) + 1) // 2 for col in support if col > 0} | {n}
    refit_cols = sorted(
        ({0} if 0 in support else set())
        | {col for freq in frequencies for col in (2 * freq - 1, 2 * freq)}
    )
    if len(refit_cols) >= M:
        raise ValueError(
            "Compressed-sensing spectrum needs more phase samples for an "
            "overdetermined refit; increase num_phases."
        )
    design = dictionary[:, refit_cols]
    refit, _, rank, _ = np.linalg.lstsq(design, signal, rcond=None)
    if rank < len(refit_cols):
        raise ValueError(
            "Compressed-sensing phase samples cannot resolve the recovered "
            "frequencies; increase num_phases or use a different seed."
        )
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

    # Propagate each phase's binomial variance through the actual refit. The
    # random design is not orthogonal, and fitting nuisance frequencies can
    # amplify shot noise substantially compared with a uniform phase grid.
    inverse = np.linalg.pinv(design)
    covariance = (inverse * np.asarray(parity_vars)) @ inverse.T
    coherence_indices = [refit_cols.index(2 * n - 1), refit_cols.index(2 * n)]
    coherence_covariance = covariance[np.ix_(coherence_indices, coherence_indices)]
    quadratures = coeffs[[2 * n - 1, 2 * n]]
    if coherence > 1e-12:
        gradient = quadratures / coherence
        coherence_var = float(gradient @ coherence_covariance @ gradient)
    else:
        # The amplitude gradient is undefined at zero; use the largest
        # quadrature variance as a conservative scale for the noise floor.
        coherence_var = float(np.linalg.eigvalsh(coherence_covariance)[-1])
    c_err = math.sqrt(max(0.0, coherence_var))

    # Phase of the frequency-n component (see docstring for the convention).
    # Undefined when the amplitude is within noise of zero: atan2 of noise.
    phase_offset: float | None = None
    phase_offset_err: float | None = None
    if coherence > max(2.0 * c_err, 1e-9):
        phase_offset = float(np.arctan2(-coeffs[2 * n], coeffs[2 * n - 1]))
        phase_gradient = np.array([quadratures[1], -quadratures[0]]) / coherence**2
        phase_var = float(phase_gradient @ coherence_covariance @ phase_gradient)
        phase_offset_err = math.sqrt(max(0.0, phase_var))

    return population, coherence, p_err, c_err, recovered_frequency, phase_offset, phase_offset_err


def estimate_fidelity_for_method(
    method: str,
    counts: list[dict[str, int]],
    phases: list[float],
    n: int,
    stabilizers: list[list[int]] | None = None,
) -> tuple[float, float, float, float, int | None, float | None, float | None]:
    """Estimate (population, coherence, p_err, c_err, recovered_frequency,
    phase_offset, phase_offset_err) for one size.

    ``counts`` holds the measurement counts for this size's circuits in
    dispatch order (Z-basis circuit first, then the method's other circuits).
    ``stabilizers`` carries the sampled Y-basis supports for dfe and is
    ignored by other methods. ``recovered_frequency`` and the phase fields are
    None for methods other than compressed sensing.
    """
    if method == "two_setting_bound":
        pop, coh, p_err, c_err = estimate_fidelity_two_setting_bound(counts[0], counts[1], n)
        return pop, coh, p_err, c_err, None, None, None
    if method == "dfe":
        pop, coh, p_err, c_err = estimate_fidelity_dfe(
            counts[0], list(counts[1:]), stabilizers or [], n
        )
        return pop, coh, p_err, c_err, None, None, None
    if method == "compressed_sensing":
        return estimate_fidelity_compressed_sensing(counts[0], list(counts[1:]), phases, n)
    pop, coh, p_err, c_err = estimate_fidelity_oscillation(counts[0], list(counts[1:]), phases, n)
    return pop, coh, p_err, c_err, None, None, None
