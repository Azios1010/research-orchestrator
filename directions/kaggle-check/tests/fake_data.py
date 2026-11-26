"""Synthetic data with the same file format as the real data. Replace along with the pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_dataset(out_dir: Path, n_subjects: int, effect: float = 0.1, seed: int = 0) -> Path:
    """One CSV per subject; method a is better than b by `effect` on average."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n_subjects):
        base = rng.uniform(0.3, 0.7)
        frame = pd.DataFrame(
            {"a": base + effect + rng.normal(0, 0.02, 5), "b": base + rng.normal(0, 0.02, 5)}
        )
        frame.to_csv(out_dir / f"subject_{index:03d}.csv", index=False)
    return out_dir


def make_dryrun_config(work_dir: Path, base_config: dict) -> dict:
    """Used by `dispatch --dry-run`: the experiment config pointed at fresh fake data."""
    data_dir = make_dataset(Path(work_dir) / "data", n_subjects=8)
    params = {**base_config["params"], "data_dir": str(data_dir), "n_subjects": 8, "n_boot": 500}
    return {"compute": dict(base_config["compute"]), "params": params}
