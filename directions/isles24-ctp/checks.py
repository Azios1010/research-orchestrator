"""Sanity checks for a finished ISLES'24 run. Only labcore's dependencies are available here."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Integer ratios, so the boundary cases are exact.
MIN_USABLE = (4, 5)  # usable cases must be at least 80% of the requested ones
MAX_GRID_MISMATCH = (1, 5)  # more than 20% grid-mismatch cases invalidates the run


def check(result_dir: Path) -> list[str]:
    result_dir = Path(result_dir)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    cases = pd.read_csv(result_dir / "cases.csv")
    rows = pd.read_csv(result_dir / "per_patient.csv")
    arms, n_folds = summary["config"]["arms"], summary["config"]["n_folds"]
    requested = summary["n_requested"]
    status = cases["status"].astype(str)
    usable = int((status == "ok").sum())
    problems = []

    if len(cases) != requested:
        problems.append(f"cases.csv has {len(cases)} rows, {requested} cases were requested")
    if usable * MIN_USABLE[1] < requested * MIN_USABLE[0]:
        download = int(status.str.startswith("download failed").sum())
        mostly = download * 2 > requested - usable
        detail = f"mostly download failed ({download})" if mostly else f"fetch errors: {download}"
        problems.append(f"usable cases {usable}/{requested} is below 80%; {detail}")
    mismatch = int(status.str.startswith("grid mismatch").sum())
    if mismatch * MAX_GRID_MISMATCH[1] > requested * MAX_GRID_MISMATCH[0]:
        problems.append(f"grid mismatch in {mismatch}/{requested} cases (more than 20%): abort")

    present = set(zip(rows["arm"], rows["fold"], strict=True))
    for arm in arms:
        for fold in range(n_folds):
            if (arm, fold) not in present:
                problems.append(f"missing results for arm {arm}, fold {fold}")
    if rows.duplicated(["subject_id", "arm"]).any():
        problems.append("a patient appears more than once for the same arm")
    per_patient = rows.groupby("subject_id")["arm"].nunique()
    if len(per_patient) != usable or (per_patient != len(arms)).any():
        problems.append("not every usable patient has a result for every arm")
    if rows["dice"].isna().any() or not rows["dice"].between(0, 1).all():
        problems.append("Dice values are missing or outside [0, 1]")
    for arm in arms:
        dice = rows.loc[rows["arm"] == arm, "dice"]
        if len(dice) and (dice == 0).all():
            problems.append(f"Dice is zero for every patient in arm {arm}")
    return problems
