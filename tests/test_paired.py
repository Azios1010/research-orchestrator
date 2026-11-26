"""Paired statistics against answers known in advance."""

import math

import numpy as np
import pytest

from labcore.stats.paired import mde, paired_summary


def test_constant_difference():
    s = paired_summary([0.6, 0.7, 0.8], [0.5, 0.6, 0.7], n_boot=500)
    assert s["n"] == 3
    assert s["mean_delta"] == pytest.approx(0.1)
    assert (s["ci_lo"], s["ci_hi"]) == (pytest.approx(0.1), pytest.approx(0.1))
    assert s["sd_delta"] == pytest.approx(0.0, abs=1e-12)
    assert s["share_a_better"] == 1.0


def test_one_two_three():
    s = paired_summary([2, 4, 6], [1, 2, 3], n_boot=2000, mde_n=49)
    assert s["mean_delta"] == 2.0
    assert s["sd_delta"] == 1.0
    assert s["mde"] == pytest.approx(2.8 * 1.0 / 7)  # 0.4
    assert 1.0 <= s["ci_lo"] < 2.0 < s["ci_hi"] <= 3.0


def test_two_point_distribution_has_exact_interval():
    # Resampled means of {0, 1} are 0, 0.5, 1 with probability 1/4, 1/2, 1/4.
    s = paired_summary([0.0, 1.0], [0.0, 0.0], n_boot=20000)
    assert (s["ci_lo"], s["ci_hi"]) == (0.0, 1.0)
    assert s["share_a_better"] == 0.5  # ties do not count as "better"


def test_mde_formula():
    assert mde(0.1, 149) == pytest.approx(2.8 * 0.1 / math.sqrt(149))
    assert mde(1.0, 49) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        mde(1.0, 0)


def test_pairs_with_a_missing_side_are_dropped():
    s = paired_summary([1.0, np.nan, 3.0, 4.0], [0.0, 1.0, np.nan, 1.0], n_boot=100)
    assert s["n"] == 2
    assert s["mean_delta"] == 2.0


def test_shared_random_stream_matches_a_direct_computation():
    rng = np.random.default_rng(5)
    a, b, c = rng.random(12), rng.random(12), rng.random(12)
    shared = np.random.default_rng(2030)
    first = paired_summary(a, b, n_boot=1000, rng=shared)
    second = paired_summary(a, c, n_boot=1000, rng=shared)
    direct = np.random.default_rng(2030)
    for summary, d in ((first, a - b), (second, a - c)):
        means = d[direct.integers(0, 12, size=(1000, 12))].mean(1)
        assert [summary["ci_lo"], summary["ci_hi"]] == list(np.percentile(means, [2.5, 97.5]))


def test_invalid_input():
    with pytest.raises(ValueError):
        paired_summary([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        paired_summary([np.nan], [1.0])
