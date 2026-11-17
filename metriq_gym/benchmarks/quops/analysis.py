"""Physical-qubit QUOPS analysis following arXiv:2609.12146v1.

The effective polarization, MCFE estimate, normal confidence bound, gated
score, and operational rate follow Supplementary Eqs. (39), (53), (62),
(63)--(65), and (72), respectively. The paired circuit bootstrap also follows
the public CUDA-Q reference implementation in Sandia's PRAQTICE repository:
https://github.com/sandialabs/PRAQTICE/tree/quops/quops/implementations/cudaq

Confidence bounds use the paper's normal approximation. They quantify sampling
uncertainty, not MCFE model bias, and are not exact finite-sample guarantees.
In particular, identical observed samples have zero bootstrap uncertainty.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from statistics import NormalDist

import numpy as np


QUOPS_THRESHOLD = 1 / math.sqrt(math.e)


@dataclass(frozen=True)
class PolarizationEstimate:
    """MCFE point estimate and its one-sided confidence bound.

    ``None`` indicates an unresolved reference denominator. A valid point
    estimate may still have unavailable uncertainty when bootstrap reference
    means cross zero. Such estimates must not pass a QUOPS test.
    """

    value: float | None
    standard_error: float | None
    lower_bound: float | None


@dataclass(frozen=True)
class ShapeEstimate:
    width: int
    size: int
    polarization: PolarizationEstimate


def effective_polarization(counts: Mapping[str, int], target: str) -> float:
    """Estimate a mirror circuit's polarization from Hamming-distance weights.

    ``target`` and the count keys must use the same qubit order. Spaces in
    count keys are ignored to support Qiskit classical-register separators.
    The estimator uses every output, rather than only the target's frequency.
    """
    if not target or set(target) - {"0", "1"}:
        raise ValueError("target must be a nonempty binary string")
    shots = 0
    weighted_sum = 0.0
    for bitstring, count in counts.items():
        bitstring = bitstring.replace(" ", "")
        if len(bitstring) != len(target) or set(bitstring) - {"0", "1"}:
            raise ValueError("count keys must be binary strings matching the target width")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("counts must be nonnegative integers")
        distance = sum(actual != expected for actual, expected in zip(bitstring, target))
        weighted_sum += count * (-0.5) ** distance
        shots += count
    if shots == 0:
        raise ValueError("counts must contain at least one shot")
    uniform_moment = 4.0 ** -len(target)
    return float((weighted_sum / shots - uniform_moment) / (1 - uniform_moment))


def normal_lower_bound(value: float, standard_error: float, *, confidence: float = 0.95) -> float:
    """Return the paper's one-sided normal bound, without clipping the estimate."""
    if not math.isfinite(confidence) or not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if not math.isfinite(standard_error) or standard_error < 0:
        raise ValueError("standard_error must be finite and nonnegative")
    return value - NormalDist().inv_cdf(confidence) * standard_error


def estimate_polarization(
    mixed: Sequence[float],
    reference: Sequence[float],
    spam: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int | None = None,
) -> PolarizationEstimate:
    """Estimate mean process polarization from the M1, M2, and M3 ensembles.

    Each mixed/reference pair belongs to the same independently sampled QUOPS
    circuit. When multiple mirrors are used, callers first average each
    circuit's mirrors. Pairing is preserved when resampling circuits; the
    width-only SPAM ensemble is resampled independently. These observed
    circuit values already include finite-shot variation, so shots and mirrors
    are not resampled a second time.

    The ratio of ensemble means is the paper's approximate MCFE estimator.
    Invalid reference means in any bootstrap draw leave uncertainty and the
    lower bound unavailable, rather than selectively discarding those draws.
    """
    normal_lower_bound(0.0, 0.0, confidence=confidence)
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, Integral)
        or bootstrap_samples < 2
    ):
        raise ValueError("bootstrap_samples must be an integer of at least two")
    ensembles = [np.asarray(values, dtype=float) for values in (mixed, reference, spam)]
    if any(values.ndim != 1 or len(values) < 2 for values in ensembles):
        raise ValueError("each ensemble must contain at least two circuit polarizations")
    if any(not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1) for values in ensembles):
        raise ValueError("circuit polarizations must be finite and between -1 and 1")
    mixed_values, reference_values, spam_values = ensembles
    if len(mixed_values) != len(reference_values):
        raise ValueError("mixed and reference ensembles must contain paired circuits")
    mean_reference = float(reference_values.mean())
    mean_spam = float(spam_values.mean())
    denominator_squared = mean_reference * mean_spam
    if mean_reference <= 0 or mean_spam <= 0 or denominator_squared == 0:
        return PolarizationEstimate(None, None, None)
    # Approximate MCFE (Supplementary Eq. 53): gamma = mean(M1) / sqrt(mean(M2) * mean(M3)).
    value = float(mixed_values.mean() / math.sqrt(denominator_squared))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(mixed_values), size=(bootstrap_samples, len(mixed_values)))
    sampled_mixed = mixed_values[indices].mean(axis=1)
    sampled_reference = reference_values[indices].mean(axis=1)
    spam_indices = rng.integers(0, len(spam_values), size=(bootstrap_samples, len(spam_values)))
    sampled_spam = spam_values[spam_indices].mean(axis=1)
    sampled_denominators_squared = sampled_reference * sampled_spam
    if (
        np.any(sampled_reference <= 0)
        or np.any(sampled_spam <= 0)
        or np.any(sampled_denominators_squared == 0)
    ):
        return PolarizationEstimate(value, None, None)
    samples = sampled_mixed / np.sqrt(sampled_denominators_squared)
    with np.errstate(over="ignore"):
        standard_error = float(samples.std(ddof=1))
    if not math.isfinite(standard_error):
        return PolarizationEstimate(value, None, None)
    return PolarizationEstimate(
        value,
        standard_error,
        normal_lower_bound(value, standard_error, confidence=confidence),
    )


def gated_score(
    rows: Sequence[ShapeEstimate], *, threshold: float = QUOPS_THRESHOLD
) -> tuple[int, tuple[int, int] | None]:
    """Certify a predeclared increasing-size sequence, stopping at its first failure.

    Every shape must satisfy the QUOPS cone ``width**2 <= size <= width**3``.
    The caller must declare the order independently of these measurements;
    sorting or selecting favorable rows after seeing results is invalid.
    Later measurements may remain useful diagnostics but cannot raise this
    score once a test fails. Confidence is inherited from the shape tests.
    """
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    previous_size = 0
    for row in rows:
        if (
            isinstance(row.width, bool)
            or not isinstance(row.width, Integral)
            or row.width < 1
            or isinstance(row.size, bool)
            or not isinstance(row.size, Integral)
            or not row.width**2 <= row.size <= row.width**3
        ):
            raise ValueError("each shape must have integer width and size within the QUOPS cone")
        if row.size <= previous_size:
            raise ValueError("the predeclared sequence must have strictly increasing sizes")
        previous_size = row.size
    score, shape = 0, None
    for row in rows:
        lower = row.polarization.lower_bound
        if lower is None or not math.isfinite(lower) or lower < threshold:
            break
        score, shape = row.size, (row.width, row.size)
    return score, shape


def quops_rate(size: int, polarization: float, total_shots: int, elapsed_seconds: float) -> float:
    """Return physical QUOPS/s without postselection: ``2*s*gamma**2*N/t``.

    ``total_shots`` counts both M1 and M2 executions. Elapsed time covers their
    execution, including switching circuits; M3 time is excluded. Queue and
    data-transfer time should be excluded when available. The factor of two
    accounts for the mirror circuits' approximately doubled depth. A simulator
    call's elapsed time produces simulator throughput, not a hardware rate.
    """
    for name, number in (("size", size), ("total_shots", total_shots)):
        if isinstance(number, bool) or not isinstance(number, Integral) or number <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(polarization) or polarization < 0:
        raise ValueError("polarization must be finite and nonnegative")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    return 2 * size * polarization**2 * total_shots / elapsed_seconds
