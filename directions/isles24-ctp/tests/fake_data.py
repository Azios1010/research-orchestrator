"""Synthetic ISLES-like shards: same parquet schema and NIfTI bytes cells, no real patients.

One parquet file per "patient", one row, columns subject_id, ncct, cta, ctp, cbf, cbv, mtt,
tmax, lesion_mask; each image cell is a struct {bytes, path} like the Hugging Face mirror.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import nibabel as nib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SHAPE = (96, 80, 16)  # X, Y, Z: deliberately not square
N_FRAMES = 24
ZOOMS = (2.0, 2.0, 5.0)
GZIPPED = ("ncct", "ctp", "lesion_mask")  # exercise both compressed and plain NIfTI cells
CELL = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def nifti_bytes(array: np.ndarray, compress: bool) -> bytes:
    zooms = ZOOMS + (1.0,) * (array.ndim - 3)
    image = nib.Nifti1Image(array, np.diag((*ZOOMS, 1.0)))
    image.header.set_zooms(zooms)
    raw = image.to_bytes()
    return gzip.compress(raw, compresslevel=1) if compress else raw


def make_case(
    rng: np.random.Generator, grid_mismatch: bool = False, empty_lesion: bool = False
) -> dict[str, np.ndarray]:
    """An elliptic 'brain' in air with a large, obvious 'lesion' that arrives late."""
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij")
    brain = (x / 0.8) ** 2 + (y / 0.8) ** 2 <= 1
    side = rng.choice([-1.0, 1.0])
    centre_y = rng.uniform(-0.15, 0.15)
    lesion = brain & (((x - side * 0.35) / 0.38) ** 2 + ((y - centre_y) / 0.45) ** 2 + (z / 0.95) ** 2 <= 1)
    if empty_lesion:  # a follow-up scan without a visible infarct
        lesion = np.zeros_like(lesion)

    def tissue(brain_value, lesion_value, noise):
        volume = np.where(brain, brain_value, 0.0) + rng.normal(0, noise, SHAPE) * brain
        volume[lesion] = lesion_value + rng.normal(0, noise, int(lesion.sum()))
        return volume

    ncct = tissue(40.0, 20.0, 2.0)
    ncct[~brain] = -1000.0
    cta = tissue(60.0, 25.0, 3.0)
    cta[~brain] = -1000.0

    t = np.arange(N_FRAMES, dtype=float)
    def bolus(arrival):
        s = np.clip(t - arrival, 0, None)
        return (s / 4.0) ** 2 * np.exp(2 - s / 2.0)

    healthy, delayed = bolus(4.0), 0.3 * bolus(8.0)
    baseline = np.where(brain, 35.0, -1000.0)
    ctp = np.repeat(baseline[..., None], N_FRAMES, axis=3)
    ctp[brain & ~lesion] += 25.0 * healthy
    ctp[lesion] += 25.0 * delayed
    ctp += rng.normal(0, 1.0, ctp.shape) * brain[..., None]

    mask = lesion.astype(np.uint8)
    arrays = {
        "ncct": ncct.round().astype(np.int16),
        "cta": cta.round().astype(np.int16),
        "ctp": ctp.round().astype(np.int16),
        "cbf": tissue(50.0, 8.0, 3.0).clip(0).astype(np.float32),
        "cbv": tissue(4.0, 1.0, 0.3).clip(0).astype(np.float32),
        "mtt": tissue(5.0, 14.0, 0.5).clip(0).astype(np.float32),
        "tmax": tissue(2.0, 12.0, 0.5).clip(0).astype(np.float32),
        "lesion_mask": mask,
    }
    if grid_mismatch:  # CTP and maps left in a different ("native CTP") grid
        for key in ("ctp", "cbf", "cbv", "mtt", "tmax"):
            arrays[key] = arrays[key][:64, :64, :12]
    return arrays


def write_shard(path: Path, subject_id: str, arrays: dict[str, np.ndarray]) -> None:
    columns = {"subject_id": pa.array([subject_id])}
    for name, array in arrays.items():
        cell = {"bytes": nifti_bytes(array, name in GZIPPED), "path": f"{name}.nii"}
        columns[name] = pa.array([cell], type=CELL)
    pq.write_table(pa.table(columns), path)


# Voxels set by add_nonfinite, per modality: (index, value). CTP indices end with the frame.
NONFINITE_VOXELS = {
    "ctp": [((48, 40, 8, 0), np.nan), ((48, 41, 8, 12), np.inf), ((49, 40, 8, 23), -np.inf)],
    "cbf": [((48, 40, 8), np.nan), ((48, 41, 8), np.inf)],
    "tmax": [((49, 40, 8), -np.inf)],
}


def add_nonfinite(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A known failure case: NaN and infinities inside the brain, in the CTP and two maps."""
    for name, voxels in NONFINITE_VOXELS.items():
        arrays[name] = arrays[name].astype(np.float32)
        for index, value in voxels:
            arrays[name][index] = value
    return arrays


def make_dataset(
    out_dir: Path,
    n_shards: int,
    grid_mismatch: tuple[int, ...] = (),
    seed: int = 0,
    nonfinite: tuple[int, ...] = (),
    empty_lesion: tuple[int, ...] = (),
) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(n_shards):
        path = out_dir / f"train-{index:05d}-of-{n_shards:05d}.parquet"
        arrays = make_case(rng, grid_mismatch=index in grid_mismatch, empty_lesion=index in empty_lesion)
        if index in nonfinite:
            arrays = add_nonfinite(arrays)
        write_shard(path, f"sub-fake{index:04d}", arrays)
    return out_dir


def smoke_params(base: dict, shard_dir: Path, cache_dir: Path, n_shards: int = 6) -> dict:
    """The tiny L0 setting: 6 cases, 2 epochs, 64 px, 3 folds. Batch, width and learning rate are
    changed only so that two epochs on a CPU already predict some lesion voxels."""
    return {
        **base,
        "local_shard_dir": str(shard_dir),
        "cache_dir": str(cache_dir),
        "n_total_shards": n_shards,
        "n_cases": n_shards,
        "epochs": 2,
        "img_size": 64,
        "n_folds": 3,
        "batch_size": 8,
        "base_ch": 8,
        "lr": 0.01,
        "n_boot": 2000,
        "save_sanity_png": False,
    }


def make_dryrun_config(work_dir: Path, base_config: dict) -> dict:
    """Used by `dispatch --dry-run`: the experiment config pointed at fresh fake shards."""
    work_dir = Path(work_dir)
    shards = make_dataset(work_dir / "shards", 6)
    params = smoke_params(base_config["params"], shards, work_dir / "cache")
    return {"compute": dict(base_config["compute"]), "params": params}
