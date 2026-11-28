"""Entrypoint: fetch and check everything -> frozen features -> two arms on patient folds -> tables.

Only aggregates are written. A patient appears as a random `patient_index` drawn from the operating
system's entropy at run time; the mapping is not saved. No folder name, file name, per-image label or
per-image prediction is printed or written.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from . import backbone as bb
from . import data as d
from . import evaluate as ev
from .experiment import run_experiment
from .probe import l2_normalise

DEFAULTS = {
    "backbone": "dinobloom-s",  # "tiny": offline testing only, with local zips
    "fold_seed": 4101,
    "split_seeds": [11, 12, 13],
    "boot_seed": 4102,
    "n_folds": 5,
    "probe_c": 1.0,
    "probe_class_weight": "balanced",
    "probe_max_iter": 5000,
    "probe_tol": 0.0001,
    "knn_k": 20,
    "n_boot": 10000,
    "batch_size": 256,
    "num_workers": 4,
    "expected_patients": 106,
    "expected_conflicting_groups": 33,
    "expected_deepapl": {},  # class -> [images, patients] after the duplicate rules
    "expected_acevedo": {},  # folder -> images
    "cache_dir": "",  # default: /kaggle/temp on Kaggle, else the system temp folder
    "min_free_gb": 3,
    "deepapl_zip": "",  # offline testing only
    "acevedo_zip": "",  # offline testing only
}
PATIENT_COLUMNS = ["patient_index", "split_seed", "arm", "class_name", "tp", "fp", "fn"]


def load_params(config_path: str) -> dict:
    params = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["params"]
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config parameters: {unknown}")
    cfg = {**DEFAULTS, **params}
    if cfg["backbone"] not in bb.BACKBONES:
        raise ValueError(f"backbone must be one of {bb.BACKBONES}")
    local = [bool(cfg["deepapl_zip"]), bool(cfg["acevedo_zip"])]
    if any(local) != all(local):
        raise ValueError("deepapl_zip and acevedo_zip are for offline testing: give both or neither")
    if cfg["backbone"] == "tiny" and not all(local):
        raise ValueError("the tiny backbone is for offline testing with local zips only")
    if len(set(cfg["split_seeds"])) != len(cfg["split_seeds"]) or not cfg["split_seeds"]:
        raise ValueError("split_seeds must be distinct and not empty")
    if not cfg["expected_deepapl"] or not cfg["expected_acevedo"]:
        raise ValueError("expected_deepapl and expected_acevedo must list the expected counts")
    if not cfg["cache_dir"]:
        base = Path("/kaggle/temp") if Path("/kaggle").is_dir() else Path(tempfile.gettempdir())
        cfg["cache_dir"] = str(base / "wbc_patient_gap")
    return cfg


def fetch(cfg: dict) -> tuple[Path, Path, torch.nn.Module, dict, str]:
    """Every download, checksum and model load, before any heavy work."""
    cache = Path(cfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    if cfg["deepapl_zip"]:
        deepapl, acevedo, source = Path(cfg["deepapl_zip"]), Path(cfg["acevedo_zip"]), "local"
    else:
        free = shutil.disk_usage(cache).free / 1024**3
        if free < cfg["min_free_gb"]:
            raise RuntimeError(f"{free:.1f} GB free for downloads, {cfg['min_free_gb']} GB needed")
        deepapl, acevedo = d.download(d.DEEPAPL, cache), d.download(d.ACEVEDO, cache)
        source = "download"
    if cfg["backbone"] == "tiny":
        return deepapl, acevedo, bb.Tiny(), {"name": "tiny", "embed_dim": bb.Tiny.dim}, source
    weights = d.download(bb.WEIGHTS, cache)
    model = bb.load_dinobloom_s(weights, cache / "hub")
    info = {
        "name": "dinobloom-s",
        "dinov2_repo": bb.DINOV2_REPO,
        "dinov2_commit": bb.DINOV2_COMMIT,
        "weights_url": bb.WEIGHTS.url,
        "weights_sha256": d.sha256_file(weights),
        "embed_dim": bb.EMBED_DIM,
    }
    return deepapl, acevedo, model, info, source


def count_tables(cfg: dict, cells: d.Cells, folders: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """dataset_counts.csv and the differences from the expected counts."""
    found = d.class_counts(cells)
    expected = {k: list(v) for k, v in cfg["expected_deepapl"].items()}
    found_acevedo = {f: folders.count(f) for f in sorted(set(folders))}
    rows = []
    for name in d.CLASSES:
        want = expected.get(name, [None, None])
        rows.append({
            "dataset": "DeepAPL", "class_name": name, "images": found[name][0], "patients": found[name][1],
            "expected_images": want[0], "expected_patients": want[1],
        })
    for folder in sorted(set(found_acevedo) | set(cfg["expected_acevedo"])):
        rows.append({
            "dataset": "Acevedo", "class_name": folder, "images": found_acevedo.get(folder, 0),
            "patients": None, "expected_images": cfg["expected_acevedo"].get(folder),
            "expected_patients": None,
        })
    problems = d.count_problems(expected, found, "DeepAPL class")
    problems += d.count_problems(cfg["expected_acevedo"], found_acevedo, "Acevedo folder")
    n_patients = len(set(cells.patients.tolist()))
    if n_patients != cfg["expected_patients"]:
        problems.append(f"DeepAPL patients: found {n_patients}, expected {cfg['expected_patients']}")
    conflicts = cells.stats["conflicting_groups"]
    if conflicts != cfg["expected_conflicting_groups"]:
        problems.append(
            f"DeepAPL duplicate groups with conflicting classes: found {conflicts}, "
            f"expected {cfg['expected_conflicting_groups']}"
        )
    table = pd.DataFrame(rows)
    for column in ("patients", "expected_images", "expected_patients"):
        table[column] = table[column].astype("Int64")
    return table, problems


def write_outputs(out: Path, cfg: dict, result: dict, header: dict) -> None:
    probe, voted = result["probe"], result["knn"]
    n_patients = len(result["patients"])
    rng = np.random.default_rng(cfg["boot_seed"])
    weights = ev.bootstrap_weights(n_patients, cfg["n_boot"], rng)
    every = tuple(range(len(d.CLASSES)))

    contrasts = []
    quantities = [("probe", "macro_f1_primary", ev.PRIMARY_INDEX, True)]
    quantities += [("probe", f"f1 {name}", (d.CLASSES.index(name),), False) for name in ev.PRIMARY]
    quantities += [("probe", "macro_f1_all12", every, False)]
    quantities += [("knn", "macro_f1_primary", ev.PRIMARY_INDEX, False)]
    for method, metric, classes, gated in quantities:
        tables = probe if method == "probe" else voted
        stats = ev.paired_bootstrap(tables["disjoint"], tables["shared"], classes, weights)
        contrasts.append({
            "contrast": ev.CONTRAST, "method": method, "metric": metric, "gated": gated,
            "n_patients": stats["n_patients"], "mean_delta": stats["mean_delta"],
            "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"],
        })
        if gated:
            primary = stats
    pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)

    per_class = []
    for method, tables in (("probe", probe), ("knn", voted)):
        for arm in ev.ARMS:
            summed = tables[arm].sum(axis=1)  # (seeds, classes, 3)
            f1 = ev.f1_from_counts(summed).mean(axis=0)
            support = (summed[..., ev.TP] + summed[..., ev.FN]).mean(axis=0)
            for k, name in enumerate(d.CLASSES):
                per_class.append({
                    "method": method, "arm": arm, "class_name": name, "f1": f1[k],
                    "n_eval_per_seed": support[k],
                })
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)

    # A fresh, unrecorded order: the index says nothing about the folder name.
    index = np.random.default_rng().permutation(n_patients)
    rows = []
    for s, seed in enumerate(cfg["split_seeds"]):
        for p in np.argsort(index):
            for arm in ev.ARMS:
                for k, name in enumerate(d.CLASSES):
                    tp, fp, fn = (int(v) for v in probe[arm][s, p, k])
                    rows.append((int(index[p]), seed, arm, name, tp, fp, fn))
    pd.DataFrame(rows, columns=PATIENT_COLUMNS).to_csv(out / "patient_counts.csv", index=False)

    external = []
    group_f1 = {}
    for dataset, counts, probes_per_cell in (
        ("Acevedo", result["external"], cfg["n_folds"]),
        ("DeepAPL held-out", result["heldout"], 1),
    ):
        f1 = ev.f1_from_counts(counts)  # (seeds, groups)
        support = counts[..., ev.TP] + counts[..., ev.FN]
        recall = np.divide(counts[..., ev.TP], support, out=np.zeros_like(f1), where=support > 0)
        group_f1[dataset] = float(f1.mean())
        for g, name in enumerate(ev.GROUP_NAMES):
            external.append({
                "dataset": dataset, "arm": "disjoint", "group_name": name,
                "n_cells": float(support[:, g].mean() / probes_per_cell),
                "recall": float(recall[:, g].mean()), "f1": float(f1[:, g].mean()),
            })
    pd.DataFrame(external).to_csv(out / "external.csv", index=False)

    pd.DataFrame(result["folds"]).to_csv(out / "folds.csv", index=False)
    pd.DataFrame(result["split_counts"]).to_csv(out / "split_counts.csv", index=False)

    folds = result["folds"]
    evaluated = probe["disjoint"].sum(axis=2)  # (seeds, patients, 3)
    cells_scored = evaluated[..., ev.TP] + evaluated[..., ev.FN]
    shares = result["shares"]
    voted_f1 = {arm: ev.arm_f1(voted[arm], ev.PRIMARY_INDEX) for arm in ev.ARMS}
    summary = {
        "config": cfg,
        **header,
        "n_requested": cfg["expected_patients"],
        # A patient is usable if it has a cell to evaluate under every split seed.
        "n_usable": int((cells_scored > 0).all(axis=0).sum()),
        "n_patients": n_patients,
        "n_eval_cells_per_seed": float(cells_scored.sum(axis=1).mean()),
        "n_donor_cells_in_shared_train_per_fold": float(
            np.mean([row["n_train_donor"] for row in folds if row["arm"] == "shared"])
        ),
        "checks": {
            "eval_in_train": sum(row["n_eval_in_train"] for row in folds),
            "eval_hash_in_train": sum(row["n_eval_hash_in_train"] for row in folds),
            "test_patient_cells_in_disjoint_train": sum(
                row["n_test_patient_cells_in_train"] for row in folds if row["arm"] == "disjoint"
            ),
            "unconverged_probes": sum(not row["probe_converged"] for row in folds),
        },
        "arms": {
            arm: {
                "macro_f1_primary": ev.arm_f1(probe[arm], ev.PRIMARY_INDEX),
                "macro_f1_all12": ev.arm_f1(probe[arm], every),
                "knn_macro_f1_primary": voted_f1[arm],
            }
            for arm in ev.ARMS
        },
        "contrasts": {ev.CONTRAST: primary},
        "neighbours": {
            "k": cfg["knn_k"],
            "same_patient_share": float(shares[0] / shares[3]),
            "same_patient_share_at_random": float(shares[1] / shares[3]),
            "same_patient_share_at_random_same_class": float(shares[2] / shares[3]),
        },
        "external": {
            "acevedo_macro_f1_7": group_f1["Acevedo"],
            "deepapl_heldout_macro_f1_7": group_f1["DeepAPL held-out"],
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(
        f"{ev.CONTRAST}: {primary['mean_delta']:+.4f} [{primary['ci_lo']:+.4f}, {primary['ci_hi']:+.4f}] "
        f"over {n_patients} patients",
        flush=True,
    )


def run(config_path: str, out_dir: str) -> None:
    started = time.time()
    cfg = load_params(config_path)
    print(cfg, flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. everything that can fail cheaply
    deepapl, acevedo, model, backbone_info, source = fetch(cfg)
    cells = d.index_deepapl(deepapl)
    external_names, folders = d.index_acevedo(acevedo)
    counts, problems = count_tables(cfg, cells, folders)
    counts.to_csv(out / "dataset_counts.csv", index=False)
    print(cells.stats, flush=True)
    if problems:
        raise RuntimeError("the data differ from the expected counts: " + "; ".join(problems))
    used = [i for i, folder in enumerate(folders) if d.ACEVEDO_FOLDERS[folder] is not None]
    external_names = [external_names[i] for i in used]
    external_groups = np.array(
        [ev.GROUP_NAMES.index(d.ACEVEDO_FOLDERS[folders[i]]) for i in used], dtype=np.int64
    )

    # 2. frozen features
    options = {"batch_size": cfg["batch_size"], "workers": cfg["num_workers"], "device": device}
    x = l2_normalise(bb.extract(model, deepapl, cells.names, dataset="DeepAPL", **options))
    x_external = l2_normalise(bb.extract(model, acevedo, external_names, dataset="Acevedo", **options))
    del model
    print(f"features done after {time.time() - started:.0f}s", flush=True)

    # 3. the two arms
    result = run_experiment(
        x, cells.labels, cells.patients, cells.hashes, x_external, external_groups, cfg
    )

    # 4. aggregates only
    versions = {"torch": torch.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__}
    header = {
        "device": device,
        "versions": versions,
        "backbone": backbone_info,
        "data": {
            "source": source,
            "deepapl_sha256": d.sha256_file(deepapl),
            "acevedo_sha256": d.sha256_file(acevedo),
            "n_external_images": len(external_names),
            **cells.stats,
        },
    }
    write_outputs(out, cfg, result, header)
    print(f"done after {time.time() - started:.0f}s", flush=True)
