"""Entrypoint: data -> folds -> every arm on every fold -> per-patient statistics."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .arms import ARMS, make_folds, n2_permutations, n_channels
from .data import load_cases
from .evaluate import eval_patients, lesion_strata, paired_statistics, primary_subjects, sanity_plot
from .model import UNet2D
from .train import device, train_one
from .zenodo import load_manifest, select_subjects

DEFAULTS = {
    "repo_id": "hugging-science/isles24-stroke",
    "n_total_shards": 149,
    "n_cases": 40,
    "seed": 2027,
    "t_window": 16,  # CTP frames kept around the bolus peak
    "t_pre": 5,  # frames kept before the peak
    "n_baseline": 3,  # first frames averaged as pre-contrast baseline
    "img_size": 128,  # in-plane size after downsampling
    "min_brain_frac": 0.02,  # keep slices with at least this brain fraction
    "n_folds": 5,
    "epochs": 40,
    "batch_size": 16,
    "lr": 1e-3,
    "base_ch": 32,
    "lesion_slice_frac": 0.5,  # share of training batches drawn from lesion slices
    "n_boot": 10000,
    "arms": list(ARMS),
    "max_grid_mismatch_frac": 0.2,  # stop before training above this
    "max_empty_gt_frac": 0.25,  # stop before training if more usable cases than this have an empty mask
    "save_sanity_png": True,
    "cache_dir": "",  # default: /kaggle/temp/cache on Kaggle, else the system temp folder
    "download_threads": 22,  # solid blocks of the archive streamed at the same time
    "prepare_threads": 2,  # subjects preprocessed at the same time
    "probe_minutes": 10,  # after this long, stop if the measured speed would pass max_fetch_hours
    "max_fetch_hours": 2.5,  # stop fetching beyond this, so a slow archive does not hold a GPU
    "min_free_gb": 30,  # free disk the cache needs: each registered CTP is about 5 GB unpacked
    "local_shard_dir": "",  # offline testing only
    "archive": "",  # offline testing only: a local file instead of the manifest's url
    "manifest": "",  # offline testing only: default is src/zenodo_manifest.json
}


def load_params(config_path: str) -> dict:
    document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    params = document["params"]
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config parameters: {unknown}")
    cfg = {**DEFAULTS, **params}
    if not cfg["cache_dir"]:
        # UNVERIFIED: /kaggle/temp has min_free_gb free (it was writable in the isles24-ctp run).
        base = Path("/kaggle/temp") if Path("/kaggle").is_dir() else Path(tempfile.gettempdir())
        cfg["cache_dir"] = str(base / "isles24_cache")
    return cfg


def check_usable(cfg: dict, chosen: list[str], meta_df: pd.DataFrame, n_usable: int) -> None:
    """Stop before spending GPU time on a run that cannot be valid."""
    statuses = meta_df["status"].astype(str)
    mismatch = int(statuses.str.startswith("grid mismatch").sum())
    if mismatch > cfg["max_grid_mismatch_frac"] * len(chosen):
        raise RuntimeError(
            f"grid mismatch in {mismatch}/{len(chosen)} cases: some volume is not on the lesion mask's "
            "grid (see the status column of cases.csv)"
        )
    if n_usable < cfg["n_folds"] * 2:
        failed = len(chosen) - n_usable
        download = int(statuses.str.startswith("download failed").sum())
        reason = f"fetch errors: {download}"
        if failed and download * 2 > failed:
            examples = statuses[statuses.str.startswith("download failed")].iloc[0][:300]
            reason = f"mostly download failed ({download} of {failed} failures), e.g. {examples}"
        raise RuntimeError(
            f"too few usable cases: {n_usable}/{len(chosen)} (need {cfg['n_folds'] * 2}); "
            f"grid mismatch: {mismatch}; {reason}"
        )


def check_empty_masks(cfg: dict, data: dict) -> None:
    """Stop before training if too many cases have no lesion left after downsampling.

    Those cases are left out of the paired statistics (evaluate.primary_subjects), so too many of
    them would make the result about a different, smaller cohort than the one requested.
    """
    empty = sum(1 for arrays, _ in data.values() if not arrays["mask"].any())
    if empty > cfg["max_empty_gt_frac"] * len(data):
        raise RuntimeError(
            f"{empty}/{len(data)} usable cases have an empty lesion mask after downsampling "
            f"(allowed: {cfg['max_empty_gt_frac']:.0%}); the paired statistics would drop them"
        )


def resume_rows(out: Path, key: str) -> list[dict]:
    """Rows already computed by an earlier, interrupted run with the same settings and patients."""
    results, marker = out / "per_patient.csv", out / "resume_key.json"
    if results.is_file() and marker.is_file() and json.loads(marker.read_text()) == key:
        return pd.read_csv(results).to_dict("records")
    marker.write_text(json.dumps(key))
    return []


def run(config_path: str, out_dir: str) -> None:
    cfg = load_params(config_path)
    print(cfg, flush=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(cfg["cache_dir"]).mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])

    # 1. patients
    manifest = load_manifest(cfg["manifest"])
    chosen = select_subjects(manifest, cfg["n_cases"])
    print("subjects:", chosen)
    data, meta_df, stopped = load_cases(cfg, manifest, chosen)
    meta_df.to_csv(out / "cases.csv", index=False)
    print(meta_df["status"].value_counts().to_string())
    if stopped:
        raise RuntimeError(stopped)
    check_usable(cfg, chosen, meta_df, len(data))
    check_empty_masks(cfg, data)

    if cfg["save_sanity_png"]:
        try:
            first = next(iter(data))
            sanity_plot(first, data[first][0], cfg, str(out / "sanity.png"))
        except Exception as e:
            print("plot skipped:", type(e).__name__, e)

    # 2. arms and folds
    sids = sorted(data)
    perms = n2_permutations(sids, cfg["seed"], cfg["t_window"])
    folds = make_folds(sids, cfg["seed"], cfg["n_folds"])
    (out / "folds.json").write_text(json.dumps(folds, indent=1), encoding="utf-8")
    print([len(f) for f in folds])

    # 3. train and evaluate (saved after every arm x fold, so a cut session can resume)
    dev = device()
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    for a in cfg["arms"]:
        model = UNet2D(n_channels(a, cfg["t_window"]), cfg["base_ch"])
        print(a, "params:", sum(p.numel() for p in model.parameters()))
    local_only = ("cache_dir", "local_shard_dir", "archive", "manifest")
    settings = {k: v for k, v in cfg.items() if k not in local_only}
    key = hashlib.sha256(json.dumps([settings, sids], sort_keys=True).encode()).hexdigest()
    all_rows = resume_rows(out, key)
    done_keys = {(r["arm"], r["fold"]) for r in all_rows}
    for f, val in enumerate(folds):
        train = [s for s in sids if s not in val]
        for arm in cfg["arms"]:
            if (arm, f) in done_keys:
                continue
            t0 = time.time()
            model = train_one(cfg, data, arm, train, perms, cfg["seed"] + 100 * f, dev)  # same seed per fold
            rows = eval_patients(model, data, arm, val, perms, dev)
            for r in rows:
                r["fold"] = f
            all_rows += rows
            pd.DataFrame(all_rows).to_csv(out / "per_patient.csv", index=False)
            mean_dice = np.mean([r["dice"] for r in rows])
            print(f"fold {f} {arm:5s}: mean Dice {mean_dice:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

    # 4. paired statistics
    write_summary(out, cfg, pd.DataFrame(all_rows), meta_df, len(sids), len(chosen), dev)


def write_summary(
    out: Path, cfg: dict, res: pd.DataFrame, meta_df, n_usable: int, n_requested: int, dev: str
):
    primary = primary_subjects(res)
    n_excluded = res["subject_id"].nunique() - primary["subject_id"].nunique()
    wide = primary.pivot_table(index="subject_id", columns="arm", values="dice")
    vol = res.pivot_table(index="subject_id", columns="arm", values="abs_vol_err_ml")
    print(f"{len(wide)} subjects in the paired statistics, {n_excluded} excluded (empty lesion mask)")
    print("Mean Dice per arm:\n", wide.mean().round(4))
    print("Median abs volume error (mL):\n", vol.median().round(2))
    stats, exact = paired_statistics(res, cfg)
    stats.to_csv(out / "paired_stats.csv", index=False)
    print(stats.to_string(index=False))
    strata = lesion_strata(res)
    strata.to_csv(out / "strata.csv", index=False)
    print("\nMean Dice by follow-up lesion size:\n", strata.to_string(index=False))
    summary = {
        "config": cfg,
        "device": dev,
        "n_requested": n_requested,
        "n_usable": n_usable,
        "n_primary": int(len(wide)),
        "n_excluded_empty_gt": int(n_excluded),
        "n_grid_mismatch": int(meta_df["status"].astype(str).str.startswith("grid mismatch").sum()),
        "mean_dice": wide.mean().round(4).to_dict(),
        "paired": stats.to_dict("records"),
        # full precision: gates are evaluated on these, not on the rounded table
        "arms": {
            arm: {
                "mean_dice": float(wide[arm].mean()),
                "median_abs_vol_err_ml": float(vol[arm].median()),
            }
            for arm in wide.columns
        },
        "contrasts": exact,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
