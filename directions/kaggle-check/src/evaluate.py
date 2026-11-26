"""Evaluation: per-subject scores and the paired statistics.

This file is covered by the pre-registration lock (with checks.py): once the direction is in L1 it
cannot change. Keep everything that turns outputs into a number a gate can see in here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from labcore.stats.paired import paired_summary


def score_subject(path: Path) -> dict:
    scores = pd.read_csv(path)
    return {"subject_id": path.stem, "a": scores["a"].mean(), "b": scores["b"].mean()}


def summarise(table: pd.DataFrame, n_requested: int, n_boot: int, seed: int) -> dict:
    contrast = paired_summary(table["a"], table["b"], n_boot=n_boot, rng=np.random.default_rng(seed))
    return {"n_requested": n_requested, "n_usable": len(table), "contrasts": {"a-b": contrast}}
