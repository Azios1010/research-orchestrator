"""Data access and preprocessing: one parquet shard per patient, NIfTI bytes in the cells.

Each shard is preprocessed to small arrays in memory, then deleted. The preprocessing is the
notebook's except for two changes: non-finite voxels in the maps and the CTP are replaced, and
in-plane downsampling is anti-aliased. These and the changes to file handling are listed in
docs/DECISIONS-bootstrap.md.
"""

from __future__ import annotations

import gc
import gzip
import os
import tempfile
import time

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage

MODALITIES = ["ncct", "cta", "ctp", "cbf", "cbv", "mtt", "tmax", "lesion_mask"]
MAPS = ["cbf", "cbv", "mtt", "tmax"]
NONFINITE_COLUMNS = [f"nonfinite_{name}" for name in ("ctp", *MAPS)] + ["nonfinite_voxels"]


def shard_name(idx: int, n_total: int) -> str:
    return f"train-{idx:05d}-of-{n_total:05d}.parquet"


def choose_shards(cfg: dict, rng: np.random.Generator) -> list[int]:
    return sorted(rng.choice(cfg["n_total_shards"], size=cfg["n_cases"], replace=False).tolist())


def fetch_shard(cfg: dict, idx: int) -> tuple[str, bool]:
    """Return (path, downloaded). Downloaded shards are deleted after preprocessing."""
    name = shard_name(idx, cfg["n_total_shards"])
    if cfg["local_shard_dir"]:  # offline testing only
        return os.path.join(cfg["local_shard_dir"], name), False
    # UNVERIFIED: the mirror is reachable, its files are data/train-XXXXX-of-00149.parquet, and
    # whether it needs a token. A Kaggle secret may not be attached to a kernel pushed by API.
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    try:  # Kaggle secret, optional
        from kaggle_secrets import UserSecretsClient

        token = token or UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    path = hf_hub_download(
        cfg["repo_id"], filename=f"data/{name}", repo_type="dataset", cache_dir=cfg["cache_dir"],
        token=token,
    )
    return path, True


def to_nifti(obj, cache_dir: str, tmp_files: list[str]):
    """Decode one NIfTI cell from the parquet row (dict with bytes, or raw bytes).

    The image is written uncompressed to disk and memory-mapped, so a large 4D CTP never
    has to sit in RAM at full size.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        b = obj.get("bytes")
        if b is None and obj.get("path") and os.path.exists(obj["path"]):
            return nib.load(obj["path"])
    else:
        b = obj
    if b is None:
        return None
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False, dir=cache_dir) as f:
        f.write(b)
        path = f.name
    tmp_files.append(path)
    return nib.load(path, mmap=True)


def cleanup_tmp(tmp_files: list[str]) -> None:
    gc.collect()  # releases memory maps, which Windows requires before deleting
    while tmp_files:
        p = tmp_files.pop()
        try:
            os.remove(p)
        except OSError as exc:
            print(f"could not delete temporary file: {type(exc).__name__}")


def area_down(x, size, is_mask=False):
    """x: (Z, H, W) -> (Z, size, size) in plane, anti-aliased. Every input goes through here.

    If H and W are whole multiples of `size`, each output pixel is the mean of its block. Otherwise
    a Gaussian prefilter (sigma = (factor - 1) / 2 per axis) is followed by linear zoom. A mask is
    resampled as a fraction and returned as bool: foreground where that fraction is >= 0.5.
    """
    z, h, w = x.shape
    values = np.asarray(x, dtype=np.float32)
    if h % size == 0 and w % size == 0:
        out = values.reshape(z, size, h // size, size, w // size).mean(axis=(2, 4))
    else:
        sigma = (0, max(h / size - 1, 0) / 2, max(w / size - 1, 0) / 2)
        smooth = ndimage.gaussian_filter(values, sigma, mode="nearest")
        # grid_mode aligns pixel areas (not pixel centres of the corners), like the block mean
        out = ndimage.zoom(smooth, (1, size / h, size / w), order=1, mode="nearest", grid_mode=True)
    return out >= 0.5 if is_mask else out


def make_finite(x: np.ndarray) -> tuple[np.ndarray, int]:
    """(array without NaN or infinity, number of voxels replaced).

    NaN becomes 0. +inf and -inf are clipped to the largest and smallest finite value of the same
    array (0 if it has none).
    """
    bad = ~np.isfinite(x)
    count = int(bad.sum())
    if count == 0:
        return x, 0
    finite = x[~bad]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    return np.nan_to_num(x, nan=0.0, posinf=high, neginf=low), count


def brain_mask_from_ncct(ncct_zyx):
    m = (ncct_zyx > 0) & (ncct_zyx < 100)
    m = ndimage.binary_opening(m, iterations=2)
    out = np.zeros_like(m)
    for z in range(m.shape[0]):  # largest component per slice, holes filled
        lab, n = ndimage.label(m[z])
        if n == 0:
            continue
        sizes = ndimage.sum(m[z], lab, range(1, n + 1))
        out[z] = ndimage.binary_fill_holes(lab == (np.argmax(sizes) + 1))
    return out


def xyz_to_zyx(a):
    # 4D -> (T, Z, Y, X)
    return np.transpose(a, (2, 1, 0)) if a.ndim == 3 else np.transpose(a, (3, 2, 1, 0))


def read_zyx(image) -> np.ndarray:
    return xyz_to_zyx(np.asarray(image.dataobj, dtype=np.float32))


def ctp_frame(ctp_px, t: int) -> tuple[np.ndarray, int]:
    """One time frame, read from the memory map, made finite; and the number of voxels replaced."""
    return make_finite(xyz_to_zyx(np.asarray(ctp_px[..., t], dtype=np.float32)))


def preprocess_case(path: str, cfg: dict, tmp_files: list[str]):
    """Return (arrays, meta), or (None, meta) with the reason in meta['status']."""
    # UNVERIFIED: one row per file, columns subject_id and one {bytes, path} cell per modality.
    tbl = pq.read_table(path)
    row = tbl.slice(0, 1).to_pylist()[0]
    sid = row.get("subject_id") or os.path.basename(path)
    imgs = {k: to_nifti(row.get(k), cfg["cache_dir"], tmp_files) for k in MODALITIES}
    del row, tbl
    missing = [k for k, v in imgs.items() if v is None]
    if missing:
        return None, {"subject_id": sid, "status": f"missing {missing}"}

    shapes = {k: tuple(v.shape) for k, v in imgs.items()}
    ref = shapes["lesion_mask"]
    bad = [k for k, s in shapes.items() if s[:3] != ref]
    if bad:
        return None, {
            "subject_id": sid,
            "status": f"grid mismatch {bad}",
            **{f"shape_{k}": str(s) for k, s in shapes.items()},
        }

    zooms = imgs["lesion_mask"].header.get_zooms()[:3]
    ncct, cta = read_zyx(imgs["ncct"]), read_zyx(imgs["cta"])
    maps, nonfinite = {}, {}
    for k in MAPS:
        maps[k], nonfinite[k] = make_finite(read_zyx(imgs[k]))
    mask = xyz_to_zyx(np.asarray(imgs["lesion_mask"].dataobj) > 0.5)

    brain = brain_mask_from_ncct(ncct)
    keep = brain.reshape(brain.shape[0], -1).mean(1) >= cfg["min_brain_frac"]
    if keep.sum() == 0:
        return None, {"subject_id": sid, "status": "no brain slices"}
    sl = np.where(keep)[0]
    S = cfg["img_size"]

    # CTP, read frame by frame from the memory map: baseline-subtracted enhancement,
    # peak-aligned window
    ctp_px = imgs["ctp"].dataobj
    T = ctp_px.shape[3]
    base = np.mean([ctp_frame(ctp_px, t)[0] for t in range(cfg["n_baseline"])], axis=0)
    curve, nonfinite["ctp"] = [], 0
    for t in range(T):  # every frame is read once here, so this is where the CTP is counted
        frame, replaced = ctp_frame(ctp_px, t)
        nonfinite["ctp"] += replaced
        curve.append((frame - base)[brain].mean())
    peak = int(np.argmax(np.array(curve)))
    start = peak - cfg["t_pre"]
    idx = np.clip(np.arange(start, start + cfg["t_window"]), 0, T - 1)  # edge padding if near the ends
    win = np.stack(
        [
            area_down(np.clip(ctp_frame(ctp_px, t)[0] - base, -20, 150)[sl] / 100.0, S).astype(np.float16)
            for t in idx
        ]
    )
    del base, ctp_px, imgs
    gc.collect()

    # maps: Tmax and MTT in seconds (fixed scale); CBF/CBV relative to the case's brain median
    def rel(x):
        med = np.median(x[brain & (x > 0)]) if np.any(brain & (x > 0)) else 1.0
        return np.clip(x / max(med, 1e-6), 0, 3) / 3.0

    m_out = np.stack(
        [
            rel(maps["cbf"]),
            rel(maps["cbv"]),
            np.clip(maps["mtt"], 0, 20) / 20.0,
            np.clip(maps["tmax"], 0, 20) / 20.0,
        ]
    )

    ncct_n = np.clip(ncct, 0, 80) / 80.0
    cta_n = np.clip(cta, -100, 400) / 500.0 + 0.2

    def down(a):
        return area_down(a[sl], S).astype(np.float16)

    out = dict(
        ncct=down(ncct_n),
        cta=down(cta_n),
        maps=np.stack([down(m) for m in m_out]),
        ctp=win,
        brain=area_down(brain[sl], S, is_mask=True),
        mask=area_down(mask[sl], S, is_mask=True),
    )
    vox_ml = float(np.prod(zooms)) / 1000.0
    scale = (mask.shape[1] / S) * (mask.shape[2] / S)
    meta = {
        "subject_id": sid,
        "status": "ok",
        "T": T,
        "peak": peak,
        "n_slices": int(len(sl)),
        "gt_ml_full": float(mask.sum() * vox_ml),
        "vox_ml_down": vox_ml * scale,
        "shape": str(shapes["lesion_mask"]),
        "zooms": str(tuple(round(float(z), 3) for z in zooms)),
        **{f"nonfinite_{name}": nonfinite[name] for name in ("ctp", *MAPS)},
        "nonfinite_voxels": sum(nonfinite.values()),
    }
    return out, meta


def load_cases(cfg: dict, chosen: list[int]) -> tuple[dict, pd.DataFrame]:
    """Fetch and preprocess every chosen shard. A failing case is recorded and skipped."""
    data, metas = {}, []
    t0 = time.time()
    for k, idx in enumerate(chosen):
        tmp_files: list[str] = []
        path, downloaded = None, False
        try:
            try:
                path, downloaded = fetch_shard(cfg, idx)
            except Exception as e:
                out, meta = None, {
                    "subject_id": f"shard{idx}",
                    "status": f"download failed: {type(e).__name__}: {e}",
                }
            else:
                out, meta = preprocess_case(path, cfg, tmp_files)
        except Exception as e:  # keep going; report at the end
            out, meta = None, {"subject_id": f"shard{idx}", "status": f"error {type(e).__name__}: {e}"}
        finally:
            cleanup_tmp(tmp_files)
            if downloaded and path and os.path.exists(path):
                os.remove(os.path.realpath(path))
        meta["shard"] = idx
        metas.append(meta)
        if out is not None:
            data[meta["subject_id"]] = (out, meta)
        replaced = meta.get("nonfinite_voxels")
        note = f", {replaced} non-finite voxels replaced" if replaced else ""
        elapsed = time.time() - t0
        print(f"[{k+1}/{len(chosen)}] shard {idx}: {meta['status']}{note}  ({elapsed:.0f}s)", flush=True)
        gc.collect()
    table = pd.DataFrame(metas)
    for column in NONFINITE_COLUMNS:  # whole numbers; empty for cases that were not preprocessed
        if column in table:
            table[column] = table[column].astype("Int64")
    return data, table
