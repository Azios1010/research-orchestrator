"""Paired per-subject statistics."""

from __future__ import annotations

import math

import numpy as np

# 2.8 = z(0.975) + z(0.80): two-sided alpha 0.05 with 80% power, normal approximation.
MDE_FACTOR = 2.8


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
