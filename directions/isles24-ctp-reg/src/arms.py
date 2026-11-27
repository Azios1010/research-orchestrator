"""Input channels per arm, the N2 frame permutation and the patient-level folds.

| arm  | channels                                                                       |
|------|--------------------------------------------------------------------------------|
| maps | NCCT, CTA, CBF, CBV, MTT, Tmax                                                 |
| raw  | NCCT, CTA, T CTP frames (peak-aligned window, baseline-subtracted)             |
| N1   | NCCT, CTA, temporal mean, temporal max (temporal axis destroyed)               |
| N2   | NCCT, CTA, the same T frames in a random order fixed per patient               |
"""

from __future__ import annotations

import numpy as np

ARMS = ("maps", "raw", "N1", "N2")


def n_channels(arm: str, t_window: int) -> int:
    return {"maps": 6, "raw": 2 + t_window, "N1": 4, "N2": 2 + t_window}[arm]


def n2_permutations(sids: list[str], seed: int, t_window: int) -> dict:
    """One fixed frame order per patient, drawn in the order of the sorted subject ids."""
    perm_rng = np.random.default_rng(seed + 1)
    return {sid: perm_rng.permutation(t_window) for sid in sids}


def arm_input(arm: str, d: dict, perm: np.ndarray) -> np.ndarray:
    """Array of shape (Z, C, H, W) for one patient."""
    base = [d["ncct"][None], d["cta"][None]]
    if arm == "maps":
        x = np.concatenate(base + [d["maps"]])
    elif arm == "raw":
        x = np.concatenate(base + [d["ctp"]])
    elif arm == "N1":
        c = d["ctp"].astype(np.float32)
        x = np.concatenate(
            base + [c.mean(0, keepdims=True).astype(np.float16), c.max(0, keepdims=True).astype(np.float16)]
        )
    elif arm == "N2":
        x = np.concatenate(base + [d["ctp"][perm]])
    else:
        raise ValueError(arm)
    return np.transpose(x, (1, 0, 2, 3))  # (Z, C, H, W)


def make_folds(sids: list[str], seed: int, n_folds: int) -> list[list[str]]:
    """Patient-level folds, identical for every arm."""
    order = np.random.default_rng(seed + 2).permutation(len(sids))
    return [[sids[i] for i in order[f::n_folds]] for f in range(n_folds)]
