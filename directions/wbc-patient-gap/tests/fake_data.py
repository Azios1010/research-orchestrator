"""Synthetic images in the real file formats: a DeepAPL-like zip and an Acevedo-like zip.

Each image is an 8 x 8 grid of colour blocks. It is the sum of a pattern of its class, a pattern of its
(patient, class) pair scaled by `patient_effect`, and noise. With a positive patient_effect, train cells
of the same patient help the probe (positive control); with 0 they cannot (negative control).

DeepAPL layout, as in the figshare archive: Patient_XX/[folder/]Signed slides/<class>/<name>.jpg, with
"Unsigned slides" copies, Thumbs.db files, a class outside the twelve, one repeated image and two images
that sit under two classes.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = (
    "Blast, no lineage spec", "Promyelocyte", "Myelocyte", "Metamyelocyte", "Band neutrophils",
    "Segmented neutrophils", "Lymphocyte", "Lymphocyte, variant", "Monocyte", "Eosinophils", "Basophil",
    "Erythroblast",
)
OTHER_CLASS = "Smudge cells"
# Acevedo folder -> the DeepAPL class whose pattern its fake images carry (None: noise only).
ACEVEDO = {
    "neutrophil": "Segmented neutrophils", "eosinophil": "Eosinophils", "basophil": "Basophil",
    "lymphocyte": "Lymphocyte", "monocyte": "Monocyte", "ig": "Myelocyte",
    "erythroblast": "Erythroblast", "platelet": None,
}
ROOT = "PBC_dataset_normal_DIB"
GRID, BLOCK = 8, 8
CLASS_STRENGTH, NOISE = 0.25, 1.0


def _jpg(values: np.ndarray) -> bytes:
    pixels = np.clip(128 + 40 * values, 0, 255).astype(np.uint8)
    pixels = pixels.repeat(BLOCK, axis=0).repeat(BLOCK, axis=1)
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def make_dataset(
    out_dir: Path,
    n_patients: int = 15,
    cells_per_class: int = 6,
    patient_effect: float = 1.0,
    seed: int = 0,
    acevedo_images: int = 6,
) -> dict:
    """Write deepapl.zip and acevedo.zip; return their paths and the counts the pipeline must find."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (GRID, GRID, 3)
    patterns = {name: rng.choice([-1.0, 1.0], shape) for name in CLASSES}

    def image(name: str | None, own: np.ndarray | float = 0.0) -> bytes:
        signal = CLASS_STRENGTH * patterns[name] if name else 0.0
        return _jpg(signal + own + NOISE * rng.standard_normal(shape))

    counts = {name: [0, 0] for name in CLASSES}
    deepapl = out_dir / "deepapl.zip"
    with zipfile.ZipFile(deepapl, "w") as archive:
        number = 0
        for p in range(n_patients):
            # Some patients have a folder between the patient and "Signed slides".
            base = f"Patient_{p:02d}/" + ("00000000001/" if p % 4 == 1 else "")
            for k, name in enumerate(CLASSES):
                if p == 3 and name == "Metamyelocyte":
                    continue  # not every patient has every class
                own = patient_effect * rng.choice([-1.0, 1.0], shape)
                blobs = [image(name, own) for _ in range(cells_per_class)]
                for blob in blobs:
                    number += 1
                    archive.writestr(f"{base}Signed slides/{name}/C{k}_{number:08d}.jpg", blob)
                counts[name][0] += len(blobs)
                counts[name][1] += 1
                archive.writestr(f"{base}Signed slides/{name}/Thumbs.db", b"not an image")
                archive.writestr(f"{base}Unsigned slides/C{k}_{number:08d}.jpg", blobs[-1])
                if p == 0 and k == 0:  # the same image twice in one class: kept once
                    archive.writestr(f"{base}Signed slides/{name}/C{k}_99999999.jpg", blobs[0])
                if p == 2 and name == "Metamyelocyte":  # also under another class: both dropped
                    archive.writestr(f"{base}Signed slides/{OTHER_CLASS}/S_00000001.jpg", blobs[0])
                    counts[name][0] -= 1
            archive.writestr(f"{base}Signed slides/{OTHER_CLASS}/S_{p + 100:08d}.jpg", image(None))
        twice = image(None)  # a new image under two classes: both dropped
        for name in ("Monocyte", "Basophil"):
            archive.writestr(f"Patient_00/Signed slides/{name}/X_00000002.jpg", twice)

    acevedo = out_dir / "acevedo.zip"
    with zipfile.ZipFile(acevedo, "w") as archive:
        for folder, name in ACEVEDO.items():
            for i in range(acevedo_images):
                archive.writestr(f"{ROOT}/{folder}/{folder.upper()}_{i:06d}.jpg", image(name))
        archive.writestr(f"{ROOT}/neutrophil/.DS_000001.jpg", b"not an image")
        archive.writestr(f"__MACOSX/{ROOT}/neutrophil/._NEUTROPHIL_000000.jpg", b"not an image")

    return {
        "deepapl": deepapl,
        "acevedo": acevedo,
        "expected_deepapl": counts,
        "expected_acevedo": {folder: acevedo_images for folder in ACEVEDO},
        "expected_patients": n_patients,
        "expected_conflicting_groups": 2,
    }


def smoke_params(base: dict, data: dict, cache_dir: Path) -> dict:
    """The experiment's parameters, pointed at the fake archives with the offline backbone."""
    return {
        **base,
        "backbone": "tiny",
        "deepapl_zip": str(data["deepapl"]),
        "acevedo_zip": str(data["acevedo"]),
        "expected_deepapl": data["expected_deepapl"],
        "expected_acevedo": data["expected_acevedo"],
        "expected_patients": data["expected_patients"],
        "expected_conflicting_groups": data["expected_conflicting_groups"],
        "cache_dir": str(cache_dir),
        "batch_size": 64,
        "num_workers": 2,
        "n_boot": 500,
    }


def make_dryrun_config(work_dir: Path, base_config: dict) -> dict:
    """Used by `dispatch --dry-run`: the experiment config pointed at fresh fake data."""
    work_dir = Path(work_dir)
    data = make_dataset(work_dir / "data")
    params = smoke_params(base_config["params"], data, work_dir / "cache")
    return {"compute": dict(base_config["compute"]), "params": params}
