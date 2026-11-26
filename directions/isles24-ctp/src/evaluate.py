"""Per-patient evaluation and the paired statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from labcore.stats.paired import paired_summary

from .arms import arm_input

CONTRASTS = [("raw", "maps"), ("raw", "N1"), ("raw", "N2"), ("maps", "N1")]
FULL_STUDY_N = 149  # the minimum detectable effect is always quoted at the full cohort size
MDE_COLUMN = "MDE_at_n149 (2.8*sd/sqrt149)"


@torch.no_grad()
def eval_patients(model, data: dict, arm: str, sids: list[str], perms: dict, dev: str) -> list[dict]:
    model.eval()
    rows = []
    for sid in sids:
        d, meta = data[sid]
        x = torch.from_numpy(arm_input(arm, d, perms[sid]).astype(np.float32))
        probs = []
        for i in range(0, len(x), 32):
            xb = x[i : i + 32].to(dev)
            p = (torch.sigmoid(model(xb)) + torch.sigmoid(model(xb.flip(-1))).flip(-1)) / 2  # flip TTA
            probs.append(p.cpu())
        pred = (torch.cat(probs)[:, 0].numpy() > 0.5) & d["brain"]
        gt = d["mask"]
        inter, s = (pred & gt).sum(), pred.sum() + gt.sum()
        dice = 1.0 if s == 0 else 2 * inter / s
        v = meta["vox_ml_down"]
        rows.append(
            {
                "subject_id": sid,
                "arm": arm,
                "dice": float(dice),
                "gt_ml": float(gt.sum() * v),
                "pred_ml": float(pred.sum() * v),
                "abs_vol_err_ml": float(abs(pred.sum() - gt.sum()) * v),
            }
        )
    return rows


def paired_statistics(results: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """(the notebook's table, rounded to 4 decimals; full-precision values per contrast).

    All contrasts draw from one bootstrap stream in the fixed CONTRASTS order.
    """
    wide = results.pivot_table(index="subject_id", columns="arm", values="dice")
    boot_rng = np.random.default_rng(cfg["seed"] + 3)
    rows, exact = [], {}
    for a, b in CONTRASTS:
        if a not in wide or b not in wide:
            continue
        s = paired_summary(
            wide[a].to_numpy(), wide[b].to_numpy(), n_boot=cfg["n_boot"], rng=boot_rng, mde_n=FULL_STUDY_N
        )
        rows.append(
            {
                "contrast": f"{a} - {b}",
                "n": s["n"],
                "mean_delta": s["mean_delta"],
                "ci95_lo": s["ci_lo"],
                "ci95_hi": s["ci_hi"],
                "sd_delta": s["sd_delta"],
                "share_patients_a_better": s["share_a_better"],
                MDE_COLUMN: s["mde"],
            }
        )
        exact[f"{a}-{b}"] = {
            "n": s["n"],
            "mean_delta": s["mean_delta"],
            "ci_lo": s["ci_lo"],
            "ci_hi": s["ci_hi"],
            "sd_delta": s["sd_delta"],
            "share_a_better": s["share_a_better"],
            "mde149": s["mde"],
        }
    return pd.DataFrame(rows).round(4), exact


def lesion_strata(results: pd.DataFrame) -> pd.DataFrame:
    """Mean Dice by follow-up lesion size (pilot only: the groups are very small)."""
    wide = results.pivot_table(index="subject_id", columns="arm", values="dice")
    gt_ml = results.groupby("subject_id")["gt_ml"].first()
    bins = pd.cut(gt_ml, [-0.1, 5, 30, 1e6], labels=["<5 mL", "5-30 mL", ">30 mL"])
    table = wide.groupby(bins, observed=False).mean().round(3)
    table.insert(0, "n", wide.groupby(bins, observed=False).size())
    table.index.name = "lesion_size"
    return table.reset_index()


def sanity_plot(sid: str, d: dict, cfg: dict, path: str) -> None:
    """One slice of one patient. It shows patient images, so it stays on Kaggle (never committed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    z = int(np.argmax(d["mask"].sum((1, 2)))) if d["mask"].any() else d["mask"].shape[0] // 2
    axes[0].imshow(d["ncct"][z].astype(np.float32), cmap="gray")
    axes[0].set_title(f"{sid} NCCT")
    axes[1].imshow(d["maps"][3, z].astype(np.float32))
    axes[1].set_title("Tmax")
    axes[2].imshow(d["ctp"][cfg["t_pre"], z].astype(np.float32))
    axes[2].set_title("CTP at peak")
    axes[3].imshow(d["mask"][z])
    axes[3].set_title("lesion mask")
    for a in axes:
        a.axis("off")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
