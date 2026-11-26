"""Direction-specific sanity checks, run by the watcher on a finished run.

`check` returns a list of problems; an empty list means the run is valid. Only labcore's
dependencies are available here (no torch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def check(result_dir: Path) -> list[str]:
    result_dir = Path(result_dir)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(result_dir / "per_subject.csv")
    problems = []
    requested, usable = summary["n_requested"], summary["n_usable"]
    if usable * 5 < requested * 4:
        problems.append(f"usable subjects {usable}/{requested} is below 80%")
    if len(table) != usable:
        problems.append(f"per_subject.csv has {len(table)} rows, expected {usable}")
    return problems
