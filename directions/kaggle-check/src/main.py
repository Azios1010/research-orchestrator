"""Entrypoint. Placeholder pipeline: compare two per-subject scores. Replace with the real one.

Input: <data_dir>/subject_XXX.csv with columns a, b (one row per item).
Output: per_subject.csv and summary.json in out_dir. Scoring and statistics live in evaluate.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .evaluate import score_subject, summarise


def run(config_path: str, out_dir: str) -> None:
    params = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["params"]
    data_dir, out = Path(params["data_dir"]), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(params["n_subjects"]):
        path = data_dir / f"subject_{index:03d}.csv"
        if not path.is_file():
            print(f"subject {index}: missing")
            continue
        rows.append(score_subject(path))
    if len(rows) < 2:
        raise RuntimeError(f"too few usable subjects: {len(rows)}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "per_subject.csv", index=False)
    summary = summarise(table, params["n_subjects"], params["n_boot"], params["seed"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
