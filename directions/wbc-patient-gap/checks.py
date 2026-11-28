"""Sanity checks of a finished run. Only labcore's dependencies are available here (no torch).

A run counts only if the two arms differ in exactly one thing: some train cells replaced by cells of
the evaluated patients. Every check below reads the tables the run wrote about what it really used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("disjoint", "shared")
MIN_USABLE = (4, 5)  # usable patients must be at least 80% of the expected ones (integers: exact)
N_CLASSES = 12


def _folds(folds: pd.DataFrame, cfg: dict) -> list[str]:
    problems = []
    wanted = {(f, s, a) for f in range(cfg["n_folds"]) for s in cfg["split_seeds"] for a in ARMS}
    found = list(zip(folds["fold"], folds["split_seed"], folds["arm"], strict=True))
    if sorted(found) != sorted(wanted):
        due = f"{len(found)} rows, {len(wanted)} due"
        problems.append(f"folds.csv must hold each fold x seed x arm once: {due}")
    for column, label in (
        ("n_eval_in_train", "evaluated cells are in a train set"),
        ("n_eval_hash_in_train", "images equal to an evaluated one are in a train set"),
    ):
        if (folds[column] != 0).any():
            problems.append(f"{int(folds[column].sum())} {label}")
    disjoint, shared = folds[folds["arm"] == "disjoint"], folds[folds["arm"] == "shared"]
    leaked = int(disjoint["n_test_patient_cells_in_train"].sum() + disjoint["n_train_donor"].sum())
    if leaked:
        problems.append("the disjoint train set holds cells of test patients")
    if (shared["n_train_donor"] <= 0).any():
        problems.append("a shared train set holds no donor cell")
    if (shared["n_test_patient_cells_in_train"] != shared["n_train_donor"]).any():
        problems.append("a shared train set holds test-patient cells other than the donor cells")
    for column, label in (("n_train", "train size"), ("n_eval", "eval size"), ("eval_set_hash", "eval set")):
        if (folds.groupby(["fold", "split_seed"])[column].nunique() != 1).any():
            problems.append(f"the arms differ in {label}")
    if not folds["probe_converged"].astype(bool).all():
        problems.append(f"{int((~folds['probe_converged'].astype(bool)).sum())} probe(s) did not converge")
    return problems


def _split_counts(counts: pd.DataFrame, cfg: dict) -> list[str]:
    problems = []
    rows = cfg["n_folds"] * len(cfg["split_seeds"]) * len(ARMS) * N_CLASSES
    keys = ["fold", "split_seed", "class_name"]
    if len(counts) != rows or counts.duplicated([*keys, "arm"]).any():
        problems.append(f"split_counts.csv has {len(counts)} rows, {rows} distinct rows are due")
    for column, label in (("n_train", "train count"), ("n_eval", "eval count")):
        if (counts.groupby(keys)[column].nunique() != 1).any():
            problems.append(f"the arms differ in the {label} of some class")
    return problems


def _data(summary: dict, table: pd.DataFrame) -> list[str]:
    cfg, problems = summary["config"], []
    deepapl = table[table["dataset"] == "DeepAPL"]
    if len(deepapl) != N_CLASSES:
        problems.append(f"dataset_counts.csv lists {len(deepapl)} DeepAPL classes, not {N_CLASSES}")
    for row in table.itertuples():
        if not row.images == row.expected_images:
            problems.append(f"{row.dataset} {row.class_name!r}: {row.images} images, not as expected")
        if row.dataset == "DeepAPL" and not row.patients == row.expected_patients:
            problems.append(f"{row.dataset} {row.class_name!r}: {row.patients} patients, not as expected")
    if summary["n_patients"] != cfg["expected_patients"]:
        problems.append(f"{summary['n_patients']} patients, expected {cfg['expected_patients']}")
    if summary["data"]["conflicting_groups"] != cfg["expected_conflicting_groups"]:
        problems.append("the number of duplicate groups with conflicting classes is not the expected one")
    if summary["data"]["source"] == "download" and summary["backbone"]["name"] != "dinobloom-s":
        problems.append("a run on downloaded data must use DinoBloom-S")
    return problems


def _patient_counts(table: pd.DataFrame, split_counts: pd.DataFrame, summary: dict) -> list[str]:
    cfg, problems = summary["config"], []
    n = summary["n_patients"]
    rows = n * len(cfg["split_seeds"]) * len(ARMS) * N_CLASSES
    if len(table) != rows or table.duplicated(["patient_index", "split_seed", "arm", "class_name"]).any():
        problems.append(f"patient_counts.csv has {len(table)} rows, {rows} distinct rows are due")
    if sorted(table["patient_index"].unique().tolist()) != list(range(n)):
        problems.append("patient_index does not number the patients 0..n-1")
    keys = ["split_seed", "arm", "class_name"]
    scored = (table["tp"] + table["fn"]).groupby([table[k] for k in keys]).sum()
    planned = split_counts.groupby(keys)["n_eval"].sum()
    if not scored.sort_index().equals(planned.sort_index().astype(scored.dtype)):
        problems.append("the cells scored per class are not the cells of the eval sets")
    return problems


def check(result_dir: Path) -> list[str]:
    result_dir = Path(result_dir)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(result_dir / "folds.csv")
    split_counts = pd.read_csv(result_dir / "split_counts.csv")
    problems = _folds(folds, summary["config"]) + _split_counts(split_counts, summary["config"])
    problems += _data(summary, pd.read_csv(result_dir / "dataset_counts.csv"))
    problems += _patient_counts(pd.read_csv(result_dir / "patient_counts.csv"), split_counts, summary)

    requested, usable = summary["n_requested"], summary["n_usable"]
    if usable * MIN_USABLE[1] < requested * MIN_USABLE[0]:
        problems.append(f"usable patients {usable}/{requested} is below 80%")
    contrasts = pd.read_csv(result_dir / "contrasts.csv")
    gated = contrasts[contrasts["gated"].astype(bool)]
    stated = summary["contrasts"]["shared-disjoint"]
    values = [stated[k] for k in ("mean_delta", "ci_lo", "ci_hi")]
    if not np.isfinite(values).all() or stated["ci_lo"] > stated["ci_hi"]:
        problems.append("the gated contrast or its interval is not a finite, ordered pair")
    elif len(gated) != 1 or not np.allclose(gated[["mean_delta", "ci_lo", "ci_hi"]].to_numpy()[0], values):
        problems.append("contrasts.csv and summary.json disagree on the gated contrast")
    for arm in ARMS:
        if not summary["arms"][arm]["macro_f1_primary"] > 0:
            problems.append(f"macro-F1 of the primary classes is zero in arm {arm}")
    return problems
