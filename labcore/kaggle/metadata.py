"""kernel-metadata.json for `kaggle kernels push`.

Key names and boolean handling were checked against the source of the kaggle CLI 2.2.5.
"""

from __future__ import annotations

import re
from pathlib import Path

from labcore.io import write_json

METADATA_NAME = "kernel-metadata.json"
# UNVERIFIED: "NvidiaTeslaT4" is the example given by `kaggle kernels push --help` (2.2.5, "GPU T4 x2").
# Confirm on the first real run that env.json reports a T4.
ACCELERATOR_T4 = "NvidiaTeslaT4"
MIN_SLUG, MAX_SLUG = 5, 50  # UNVERIFIED: the CLI enforces >= 5; the 50-character cap is server side.


def kernel_slug(slug: str, exp: str) -> str:
    """<slug>-<exp>, lowercase, hyphens only. The title is this same string, so it resolves to it."""
    name = re.sub(r"[^a-z0-9]+", "-", f"{slug}-{exp}".lower()).strip("-")
    if not MIN_SLUG <= len(name) <= MAX_SLUG:
        raise ValueError(
            f"kernel name {name!r} is {len(name)} characters; Kaggle needs {MIN_SLUG}-{MAX_SLUG}. "
            "Use a shorter experiment name."
        )
    return name


def build_metadata(
    *,
    username: str,
    slug: str,
    exp: str,
    code_file: str,
    dataset_sources: list[str] | None = None,
    accelerator: str | None = ACCELERATOR_T4,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username or ""):
        raise ValueError("a Kaggle username is required (set KAGGLE_USERNAME)")
    name = kernel_slug(slug, exp)
    metadata = {
        "id": f"{username.lower()}/{name}",
        "title": name,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": list(dataset_sources or []),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if accelerator:
        metadata["machine_shape"] = accelerator
    return metadata


def write_metadata(folder: Path, metadata: dict) -> Path:
    path = Path(folder) / METADATA_NAME
    write_json(path, metadata)
    return path
