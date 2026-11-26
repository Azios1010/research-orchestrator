"""Paired per-subject statistics."""

from __future__ import annotations

import math

import numpy as np

# 2.8 = z(0.975) + z(0.80): two-sided alpha 0.05 with 80% power, normal approximation.
MDE_FACTOR = 2.8


def _gamma_p(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x): series for x < a + 1, else a continued fraction."""
    if x <= 0.0:
        return 0.0
    log_front = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        for i in range(1, 100000):
            term *= x / (a + i)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return min(1.0, total * math.exp(log_front))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return max(0.0, 1.0 - math.exp(log_front) * h)


def chi2_ppf(p: float, df: int) -> float:
    """Quantile of the chi-square distribution (the inverse of its CDF), without scipy.

    Found by bisection on the CDF; accurate to about 1e-10 relative, checked against scipy in the tests.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    if df < 1:
        raise ValueError("df must be at least 1")
    lo, hi = 0.0, float(df) + 10.0 * math.sqrt(2.0 * df) + 10.0
    while _gamma_p(df / 2.0, hi / 2.0) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gamma_p(df / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * hi:
            break
    return 0.5 * (lo + hi)


def sd_upper(sd: float, n: int, confidence: float = 0.80) -> float:
    """One-sided upper confidence bound of the population sd from a sample sd on n pairs.

    sd * sqrt((n - 1) / chi2.ppf(1 - confidence, n - 1)); assumes roughly normal differences.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return float(sd * math.sqrt((n - 1) / chi2_ppf(1.0 - confidence, n - 1)))


def sd_delta_ucl80(sd: float, n: int) -> float:
    """80% upper confidence bound of the paired-difference sd. A pilot's sd is itself noisy."""
    return sd_upper(sd, n, 0.80)


def mde_ucl(sd: float, n_pilot: int, n_planned: int, confidence: float = 0.80) -> float:
    """Minimum detectable paired difference at n_planned, using the upper bound of the sd.

    mde() with the sd replaced by its upper confidence bound from a pilot of n_pilot pairs. If this is
    already small enough, a study of n_planned is powered even when the pilot sd was optimistic.
    """
    return mde(sd_upper(sd, n_pilot, confidence), n_planned)


def mde(sd: float, n: int) -> float:
    """Minimum detectable mean paired difference at sample size n."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return float(MDE_FACTOR * sd / math.sqrt(n))


def paired_summary(
    a,
    b,
    *,
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
    mde_n: int | None = None,
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Summarise per-subject differences d = a - b. Pairs with a NaN on either side are dropped.

    The confidence interval is a percentile bootstrap over subjects. Pass the same `rng` to
    successive calls to draw all contrasts from one random stream.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length, one value per subject")
    d = a - b
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        raise ValueError("no complete pairs")
    rng = rng if rng is not None else np.random.default_rng(0)
    samples = rng.integers(0, n, size=(n_boot, n))
    means = d[samples].mean(1)
    lo, hi = np.percentile(means, list(percentiles))
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    target = mde_n if mde_n is not None else n
    return {
        "n": n,
        "mean_delta": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "sd_delta": sd,
        "share_a_better": float((d > 0).mean()),
        "mde_n": target,
        "mde": mde(sd, target),
    }
