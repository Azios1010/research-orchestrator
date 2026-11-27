"""Data access and preprocessing, one subject at a time.

CTA, CTP, the four perfusion maps and the lesion mask come from the official release on Zenodo,
all co-registered to the NCCT grid (zenodo.py streams them out of the archive). The NCCT comes from
the Hugging Face mirror, whose rows also carry that lesion mask: the two masks must be equal, which
ties both sources to the same subject and grid.

Each subject is preprocessed to small arrays in memory and its files are deleted. The preprocessing
from the loaded volumes on is that of direction isles24-ctp (the notebook's, with non-finite voxels
in the maps and the CTP replaced and anti-aliased in-plane downsampling).
"""

from __future__ import annotations

import gc
import gzip
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import ndimage

from . import zenodo

MODALITIES = ["ncct", "cta", "ctp", "cbf", "cbv", "mtt", "tmax", "lesion_mask"]
MAPS = ["cbf", "cbv", "mtt", "tmax"]
NONFINITE_COLUMNS = [f"nonfinite_{name}" for name in ("ctp", *MAPS)] + ["nonfinite_voxels"]
# cases.csv always has exactly these columns (results_columns in direction.yaml lists the same).
CASE_COLUMNS = [
    "subject_id", "status", "T", "peak", "n_slices", "gt_ml_full", "vox_ml_down", "shape", "zooms",
    *NONFINITE_COLUMNS, "block", "shard",
]
AFFINE_TOLERANCE = 1e-3  # mm; the release stores the same affine in every co-registered file
MAX_UNUSABLE_FRAC = 0.2  # more unusable subjects than this invalidate the run, so stop fetching
POLL_SECONDS = 1.0


class MirrorMismatch(RuntimeError):
    """The mirror's row is not the subject it should be."""


def shard_name(idx: int, n_total: int) -> str:
    return f"train-{idx:05d}-of-{n_total:05d}.parquet"


def shard_index(manifest: dict, subject: str) -> int:
    """The mirror has one shard per subject, in the order of the subject names."""
    return sorted(manifest["subjects"]).index(subject)


def fetch_shard(cfg: dict, idx: int) -> tuple[str, bool]:
    """Return (path, downloaded). Downloaded shards are deleted after reading."""
    name = shard_name(idx, cfg["n_total_shards"])
    if cfg["local_shard_dir"]:  # offline testing only
        return os.path.join(cfg["local_shard_dir"], name), False
    # A Kaggle secret may not be attached to a kernel pushed by API; the mirror needs no token.
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


def _temporary_nii(cache_dir: str, tmp_files: list[str]) -> str:
    handle, path = tempfile.mkstemp(suffix=".nii", dir=cache_dir)
    os.close(handle)
    tmp_files.append(path)
    return path


def to_nifti(obj, cache_dir: str, tmp_files: list[str]):
    """Decode one NIfTI cell from the parquet row (dict with bytes, or raw bytes).

    The image is written uncompressed to disk and memory-mapped.
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
    path = _temporary_nii(cache_dir, tmp_files)
    Path(path).write_bytes(b)
    return nib.load(path, mmap=True)


def load_gz(source: Path, cache_dir: str, tmp_files: list[str]):
    """Decompress a .nii.gz to disk in pieces and memory-map it: a registered CTP is about 5 GB."""
    path = _temporary_nii(cache_dir, tmp_files)
    with gzip.open(source, "rb") as packed, open(path, "wb") as plain:
        shutil.copyfileobj(packed, plain, 16 << 20)
    return nib.load(path, mmap=True)


def cleanup_tmp(tmp_files: list[str]) -> None:
    gc.collect()  # releases memory maps, which Windows requires before deleting
    while tmp_files:
        p = tmp_files.pop()
        try:
            os.remove(p)
        except OSError as exc:
            print(f"could not delete temporary file: {type(exc).__name__}")


def read_mirror(path: str, subject: str, cfg: dict, tmp_files: list[str]):
    """(NCCT image, lesion mask image) from one shard; only these two columns are read."""
    # One row per file, columns subject_id and one {bytes, path} cell per modality.
    table = pq.read_table(path, columns=["subject_id", "ncct", "lesion_mask"])
    row = table.slice(0, 1).to_pylist()[0]
    if row["subject_id"] != subject:
        raise MirrorMismatch(f"the shard holds {row['subject_id']}, expected {subject}")
    ncct = to_nifti(row["ncct"], cfg["cache_dir"], tmp_files)
    mask = to_nifti(row["lesion_mask"], cfg["cache_dir"], tmp_files)
    if ncct is None or mask is None:
        raise MirrorMismatch("the shard lacks the NCCT or the lesion mask")
    return ncct, mask


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


def preprocess_case(subject: str, paths: dict, ncct_image, mirror_mask, cfg: dict, tmp_files: list[str]):
    """`paths` maps each of zenodo.KINDS to a .nii.gz file.

    Return (arrays, meta), or (None, meta) with the reason in meta['status'].
    """
    imgs = {kind: load_gz(paths[kind], cfg["cache_dir"], tmp_files) for kind in zenodo.KINDS}
    imgs["ncct"] = ncct_image

    shapes = {k: tuple(imgs[k].shape) for k in MODALITIES}
    ref, affine = shapes["lesion_mask"], imgs["lesion_mask"].affine
    bad = [
        k for k in MODALITIES
        if shapes[k][:3] != ref or not np.allclose(imgs[k].affine, affine, rtol=0, atol=AFFINE_TOLERANCE)
    ]
    if bad:
        detail = ", ".join(f"{k} {shapes[k]}" for k in bad)
        return None, {
            "subject_id": subject,
            "status": f"grid mismatch {bad}: shape or affine differs from the lesion mask {ref} ({detail})",
        }

    mask_xyz = np.asarray(imgs["lesion_mask"].dataobj) > 0.5
    if not np.array_equal(mask_xyz, np.asarray(mirror_mask.dataobj) > 0.5):
        return None, {
            "subject_id": subject,
            "status": "source mismatch: the lesion mask differs between Zenodo and the mirror",
        }

    zooms = imgs["lesion_mask"].header.get_zooms()[:3]
    ncct, cta = read_zyx(imgs["ncct"]), read_zyx(imgs["cta"])
    maps, nonfinite = {}, {}
    for k in MAPS:
        maps[k], nonfinite[k] = make_finite(read_zyx(imgs[k]))
    mask = xyz_to_zyx(mask_xyz)

    brain = brain_mask_from_ncct(ncct)
    keep = brain.reshape(brain.shape[0], -1).mean(1) >= cfg["min_brain_frac"]
    if keep.sum() == 0:
        return None, {"subject_id": subject, "status": "no brain slices"}
    sl = np.where(keep)[0]
    S = cfg["img_size"]

    # CTP, read frame by frame from the memory map: baseline-subtracted enhancement,
    # peak-aligned window
    ctp_px = imgs["ctp"].dataobj
    T = imgs["ctp"].shape[3]
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
        "subject_id": subject,
        "status": "ok",
        "T": T,
        "peak": peak,
        "n_slices": int(len(sl)),
        "gt_ml_full": float(mask.sum() * vox_ml),
        "vox_ml_down": vox_ml * scale,
        "shape": str(ref),
        "zooms": str(tuple(round(float(z), 3) for z in zooms)),
        **{f"nonfinite_{name}": nonfinite[name] for name in ("ctp", *MAPS)},
        "nonfinite_voxels": sum(nonfinite.values()),
    }
    return out, meta


def prepare_subject(cfg: dict, manifest: dict, subject: str, paths: dict):
    """Mirror shard + the extracted files -> (arrays, meta). Never raises; deletes what it read."""
    tmp_files: list[str] = []
    shard_path, downloaded = None, False
    try:
        try:
            shard_path, downloaded = fetch_shard(cfg, shard_index(manifest, subject))
        except Exception as e:
            return None, {"subject_id": subject, "status": f"download failed: {type(e).__name__}: {e}"}
        ncct, mirror_mask = read_mirror(shard_path, subject, cfg, tmp_files)
        out = preprocess_case(subject, paths, ncct, mirror_mask, cfg, tmp_files)
        del ncct, mirror_mask
        return out
    except MirrorMismatch as e:
        return None, {"subject_id": subject, "status": f"source mismatch: {e}"}
    except Exception as e:  # keep going; report at the end
        return None, {"subject_id": subject, "status": f"error {type(e).__name__}: {e}"}
    finally:
        cleanup_tmp(tmp_files)
        for path in paths.values():
            Path(path).unlink(missing_ok=True)
        if downloaded and shard_path and os.path.exists(shard_path):
            os.remove(os.path.realpath(shard_path))


def stop_reason(
    cfg: dict, statuses: list[str], n_requested: int, received: int, expected: int, elapsed: float
) -> str | None:
    """Why fetching should stop now, or None. A run that cannot be valid, or cannot finish in time,
    must not keep a GPU busy. Only a download failure is worded so that the watcher retries."""
    mismatch = sum(s.startswith("grid mismatch") for s in statuses)
    if mismatch > cfg["max_grid_mismatch_frac"] * n_requested:
        return f"grid mismatch in {mismatch}/{n_requested} cases (stopped early)"
    unusable = [s for s in statuses if s != "ok"]
    if len(unusable) > MAX_UNUSABLE_FRAC * n_requested:
        download = [s for s in unusable if s.startswith("download failed")]
        if len(download) * 2 > len(unusable):
            return (
                f"mostly download failed ({len(download)} of {len(unusable)} failures), e.g. "
                f"{download[0][:300]}"
            )
        return f"{len(unusable)}/{n_requested} cases unusable (stopped early), e.g. {unusable[0][:300]}"
    limit = cfg["max_fetch_hours"] * 3600
    if elapsed > limit:
        return f"fetching passed its limit of {cfg['max_fetch_hours']:g} h (max_fetch_hours)"
    if elapsed >= cfg["probe_minutes"] * 60 and received < expected:
        rate = max(received, 1) / max(elapsed, 1e-9)
        projected = expected / rate
        if projected > limit:
            return (
                f"the archive is too slow: {rate / 1e6:.2f} MB/s means about {projected / 3600:.1f} h "
                f"for {expected / 1e9:.0f} GB (max_fetch_hours {cfg['max_fetch_hours']:g})"
            )
    return None


def load_cases(cfg: dict, manifest: dict, chosen: list[str]) -> tuple[dict, pd.DataFrame, str | None]:
    """Stream the needed blocks in parallel and preprocess each subject as soon as it is complete.

    Return (data, case table, why fetching was stopped or None). A failing subject is recorded and
    skipped.
    """
    cache = Path(cfg["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / 1e9
    print(f"free disk for the cache: {free:.0f} GB", flush=True)
    if free < cfg["min_free_gb"]:
        raise RuntimeError(f"only {free:.0f} GB free for the cache; {cfg['min_free_gb']:g} GB are needed")
    open_range = zenodo.open_file(cfg["archive"]) if cfg["archive"] else zenodo.open_url(manifest["url"])
    wanted = zenodo.plan(manifest, chosen)
    expected = zenodo.expected_bytes(wanted)
    print(f"{len(chosen)} subjects in {len(wanted)} blocks, about {expected / 1e9:.1f} GB to stream")

    lock, stop = threading.Lock(), threading.Event()
    slots = threading.Semaphore(cfg["prepare_threads"] + 2)  # complete subjects waiting on disk
    results: dict[str, tuple] = {}
    arrived: dict[str, dict] = {name: {} for name in chosen}
    failed: set[str] = set()
    preparing, received, started = [], [0], time.time()

    def count(n: int) -> None:
        with lock:
            received[0] += n

    def record(subject: str, out, meta: dict) -> None:
        meta.update(block=manifest["subjects"][subject]["ctp"][0], shard=shard_index(manifest, subject))
        with lock:
            results[subject] = (out, meta)
            replaced = meta.get("nonfinite_voxels")
            note = f", {replaced} non-finite voxels replaced" if replaced else ""
            print(
                f"[{len(results)}/{len(chosen)}] {subject}: {meta['status']}{note}  "
                f"({time.time() - started:.0f}s, {received[0] / 1e9:.2f} GB streamed)",
                flush=True,
            )

    def prepare(subject: str, paths: dict) -> None:
        try:
            record(subject, *prepare_subject(cfg, manifest, subject, paths))
        finally:
            slots.release()

    def fetch(block: int) -> None:
        def on_file(subject: str, kind: str, path: Path) -> None:
            with lock:
                arrived[subject][kind] = path
                complete = len(arrived[subject]) == len(zenodo.KINDS) and subject not in failed
                paths = dict(arrived[subject])
            if not complete:
                return
            while not slots.acquire(timeout=POLL_SECONDS):
                if stop.is_set():
                    raise zenodo.Stopped("stopped")
            preparing.append(preparers.submit(prepare, subject, paths))

        try:
            zenodo.stream_block(
                open_range, manifest["blocks"][str(block)], wanted[block], cache, on_file,
                stop=stop, count=count,
            )
        except zenodo.Stopped:
            pass
        except zenodo.SourceError as e:  # anything else is a bug or a full disk and fails the run
            with lock:
                lost = {name for *_, name, kind in wanted[block] if kind not in arrived[name]} - failed
                failed.update(lost)
            for subject in sorted(lost):
                record(subject, None, {
                    "subject_id": subject, "status": f"download failed: {type(e).__name__}: {e}",
                })

    # the longest streams start first, so the last connections do not run alone for long
    order = sorted(wanted, key=lambda b: -max(start + size for start, size, *_ in wanted[b]))
    reason = None
    downloaders = ThreadPoolExecutor(cfg["download_threads"])
    preparers = ThreadPoolExecutor(cfg["prepare_threads"])
    try:
        fetching = [downloaders.submit(fetch, block) for block in order]
        while True:
            finished = all(f.done() for f in fetching) and all(f.done() for f in list(preparing))
            with lock:
                statuses = [meta["status"] for _, meta in results.values()]
                bytes_in = received[0]
            if finished:
                break
            if any(f.done() and f.exception() for f in [*fetching, *list(preparing)]):
                break  # a bug or a full disk: raised below, without waiting for the other streams
            reason = stop_reason(cfg, statuses, len(chosen), bytes_in, expected, time.time() - started)
            if reason:
                break
            time.sleep(POLL_SECONDS)
    finally:
        stop.set()
        downloaders.shutdown(wait=True, cancel_futures=True)
        preparers.shutdown(wait=True, cancel_futures=True)
    for future in [*fetching, *preparing]:  # surface programming errors
        if future.done() and not future.cancelled():
            future.result()
    if reason is None:  # the last subjects may have made the run invalid after the last look
        reason = stop_reason(
            {**cfg, "max_fetch_hours": float("inf")}, [m["status"] for _, m in results.values()],
            len(chosen), expected, expected, 0.0,
        )
    print(f"fetched {received[0] / 1e9:.2f} GB in {(time.time() - started) / 60:.1f} min", flush=True)

    data, metas = {}, []
    for subject in chosen:
        out, meta = results.get(subject, (None, {
            "subject_id": subject, "status": "not prepared: fetching stopped early",
            "block": manifest["subjects"][subject]["ctp"][0], "shard": shard_index(manifest, subject),
        }))
        metas.append(meta)
        if out is not None:
            data[subject] = (out, meta)
    table = pd.DataFrame(metas).reindex(columns=CASE_COLUMNS)
    for column in NONFINITE_COLUMNS:  # whole numbers; empty for cases that were not preprocessed
        table[column] = table[column].astype("Int64")
    return data, table, reason
