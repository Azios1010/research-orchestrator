"""A synthetic ISLES-like release in the two real file formats, without real patients.

- the mirror: one parquet file per subject, one row, columns subject_id, ncct, ctp, lesion_mask,
  each image cell a struct {bytes, path} like the Hugging Face mirror;
- the archive: solid raw-LZMA2 blocks of .nii.gz files laid end to end, as in the release's 7z,
  with the manifest that scripts/make_manifest.py would write for it. Other files sit between
  the wanted ones, and some subjects straddle two blocks.
"""

from __future__ import annotations

import gzip
import json
import lzma
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SHAPE = (96, 80, 16)  # X, Y, Z: deliberately not square
N_FRAMES = 24
# tilted like a gantry-tilted CT, so that an affine is more than its zooms
AFFINE = np.array([[-2.0, 0, 0, 96], [0, 1.9, -1.2, -80], [0, 0.5, 4.8, -40], [0, 0, 0, 1.0]])
KINDS = ("cbf", "cbv", "mtt", "tmax", "cta", "ctp", "lesion_mask")
DICT_SIZE = 1 << 24
CELL = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def nifti_gz(array: np.ndarray, affine: np.ndarray = AFFINE) -> bytes:
    return gzip.compress(nib.Nifti1Image(array, affine).to_bytes(), compresslevel=1, mtime=0)


def make_case(rng: np.random.Generator, empty_lesion: bool = False) -> dict[str, np.ndarray]:
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

    return {
        "ncct": ncct.round().astype(np.int16),
        "cta": cta.astype(np.float32),
        "ctp": ctp.astype(np.float32),
        "cbf": tissue(50.0, 8.0, 3.0).clip(0).astype(np.float32),
        "cbv": tissue(4.0, 1.0, 0.3).clip(0).astype(np.float32),
        "mtt": tissue(5.0, 14.0, 0.5).clip(0).astype(np.float32),
        "tmax": tissue(2.0, 12.0, 0.5).clip(0).astype(np.float32),
        "lesion_mask": lesion.astype(np.float64),
    }


# Voxels set by add_nonfinite, per file: (index, value). CTP indices end with the frame.
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


def write_shard(path: Path, subject: str, ncct: np.ndarray, mask: np.ndarray) -> None:
    """The mirror also holds the CTP in native space (another grid); the pipeline must not read it."""
    cells = {"ncct": ncct, "ctp": np.zeros((8, 8, 4, 3), np.int16), "lesion_mask": mask}
    columns = {"subject_id": pa.array([subject])}
    for name, array in cells.items():
        cell = {"bytes": nifti_gz(array), "path": f"{subject}_{name}.nii.gz"}
        columns[name] = pa.array([cell], type=CELL)
    pq.write_table(pa.table(columns), path)


def write_archive(path: Path, files: list[tuple[str, str, bytes]], per_block: int) -> dict:
    """Write (subject, kind, bytes) entries into solid blocks and return the manifest."""
    manifest = {"source": "synthetic", "url": "", "blocks": {}, "subjects": {}}
    content = bytearray(b"\0" * 32)  # where the 7z signature header would be
    for number, first in enumerate(range(0, len(files), per_block)):
        plain = bytearray()
        for subject, kind, payload in files[first : first + per_block]:
            if kind in KINDS:
                manifest["subjects"].setdefault(subject, {})[kind] = [
                    number, len(plain), len(payload), zlib.crc32(payload),
                ]
            plain += payload
        packed = lzma.compress(
            bytes(plain), format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 1}]
        )
        manifest["blocks"][str(number)] = {
            "offset": len(content), "packed": len(packed), "dict_size": DICT_SIZE,
        }
        content += packed
    manifest["archive_size"] = len(content)
    Path(path).write_bytes(bytes(content))
    return manifest


def make_dataset(
    out_dir: Path,
    n_cases: int,
    grid_mismatch: tuple[int, ...] = (),
    seed: int = 0,
    nonfinite: tuple[int, ...] = (),
    empty_lesion: tuple[int, ...] = (),
    shifted: tuple[int, ...] = (),
    source_mismatch: tuple[int, ...] = (),
    per_block: int = 11,
) -> dict[str, Path]:
    """Known failure cases: `grid_mismatch` (CTP left in another shape), `shifted` (CTA with the
    right shape but another affine) and `source_mismatch` (the mirror's mask differs)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    shards = out_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    files = []
    for index in range(n_cases):
        subject = f"sub-fake{index:04d}"
        arrays = make_case(rng, empty_lesion=index in empty_lesion)
        if index in nonfinite:
            arrays = add_nonfinite(arrays)
        if index in grid_mismatch:
            arrays["ctp"] = arrays["ctp"][:64, :64, :12]
        mirror_mask = arrays["lesion_mask"]
        if index in source_mismatch:
            mirror_mask = np.roll(mirror_mask, 7, axis=0)
        write_shard(
            shards / f"train-{index:05d}-of-{n_cases:05d}.parquet", subject, arrays["ncct"], mirror_mask
        )
        for kind in KINDS:
            affine = AFFINE
            if kind == "cta" and index in shifted:
                affine = AFFINE.copy()
                affine[0, 3] += 3.0
            files.append((subject, kind, nifti_gz(arrays[kind], affine)))
        files.append((subject, "dwi", bytes(rng.integers(0, 256, 3000, dtype=np.uint8))))
    manifest = write_archive(out_dir / "train.fake", files, per_block)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"shards": shards, "archive": out_dir / "train.fake", "manifest": manifest_path}


def smoke_params(base: dict, dataset: dict, cache_dir: Path, n_cases: int = 6) -> dict:
    """The tiny L0 setting: 6 cases, 2 epochs, 64 px, 3 folds. Batch, width and learning rate are
    changed only so that two epochs on a CPU already predict some lesion voxels."""
    return {
        **base,
        "local_shard_dir": str(dataset["shards"]),
        "archive": str(dataset["archive"]),
        "manifest": str(dataset["manifest"]),
        "cache_dir": str(cache_dir),
        "n_total_shards": n_cases,
        "n_cases": n_cases,
        "download_threads": 3,
        "min_free_gb": 0.01,
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
    """Used by `dispatch --dry-run`: the experiment config pointed at a fresh fake release."""
    work_dir = Path(work_dir)
    dataset = make_dataset(work_dir / "release", 6)
    params = smoke_params(base_config["params"], dataset, work_dir / "cache")
    return {"compute": dict(base_config["compute"]), "params": params}
