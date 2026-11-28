"""Frozen feature extractors: DinoBloom-S, and a tiny stand-in that needs no network.

DinoBloom-S is loaded as in marrlab/DinoBloom, dinov2/eval/models/return_model.py (commit 6c5e86d):
the DINOv2 ViT-S/14 from torch.hub, the checkpoint's "teacher" weights without the training heads and
without the "backbone." prefix, a 257-token position embedding, and a strict load. Nothing is trained.
"""

from __future__ import annotations

import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import DataError, Source, read_member

DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_COMMIT = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
DINOV2_ARCH = "dinov2_vits14"
# Zenodo record 10908163 (CC BY 4.0), DinoBloom-S.pth (md5 8bade579...).
WEIGHTS = Source(
    "DinoBloom-S.pth",
    "https://zenodo.org/records/10908163/files/DinoBloom-S.pth?download=1",
    265416526,
    "c2f7990b003e89bcece80e379fb8fe0ba2ec392ce19b286e8a294abd99568e44",
)
EMBED_DIM = 384
IMAGE_SIZE = 224
POSITIONS = 257  # 16 x 16 patches of a 224 pixel image, plus the class token
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BACKBONES = ("dinobloom-s", "tiny")
HEADS = ("dino_head", "ibot_head")
PREFIX = "backbone."


def preprocess(blob: bytes) -> np.ndarray:
    """JPG bytes -> (3, 224, 224): RGB, bilinear resize, ImageNet normalisation."""
    with Image.open(io.BytesIO(blob)) as image:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = (np.asarray(resized, dtype=np.float32) / 255.0 - MEAN) / STD
    return array.transpose(2, 0, 1)


def convert_teacher_state_dict(checkpoint: dict) -> dict:
    """The backbone weights of the checkpoint's teacher, under the names the hub model uses."""
    if "teacher" not in checkpoint:
        raise ValueError("the checkpoint has no 'teacher' entry")
    state = {}
    for key, value in checkpoint["teacher"].items():
        if any(head in key for head in HEADS):
            continue
        name = key.removeprefix(PREFIX)
        if name in state:
            raise ValueError(f"two checkpoint keys map to {name!r}")
        state[name] = value
    if not state:
        raise ValueError("the teacher entry holds no backbone weight")
    return state


def load_teacher(model: torch.nn.Module, checkpoint: dict) -> torch.nn.Module:
    """Give the model the 224 pixel position embedding and load every weight, strictly."""
    state = convert_teacher_state_dict(checkpoint)
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, POSITIONS, EMBED_DIM))
    model.load_state_dict(state, strict=True)
    return model


def hub_model(hub_dir: Path | None = None) -> torch.nn.Module:
    """The untrained ViT-S/14 of the pinned dinov2 commit."""
    os.environ["XFORMERS_DISABLED"] = "1"  # the same attention code on every machine
    if hub_dir is not None:
        torch.hub.set_dir(str(hub_dir))
    # skip_validation: torch.hub otherwise refuses a commit that is not the head of a branch or tag.
    # pretrained=False: Meta's weights would be overwritten by the strict load anyway.
    # UNVERIFIED: a Kaggle kernel can fetch the repository from GitHub through torch.hub.
    return torch.hub.load(
        f"{DINOV2_REPO}:{DINOV2_COMMIT}",
        DINOV2_ARCH,
        pretrained=False,
        trust_repo=True,
        skip_validation=True,
    )


def load_dinobloom_s(weights_path: Path, hub_dir: Path | None = None) -> torch.nn.Module:
    """`weights_path` must already be verified against WEIGHTS (data.verify)."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    return load_teacher(hub_model(hub_dir), checkpoint).eval()


class Tiny(torch.nn.Module):
    """Average colour on an 8 x 8 grid. For offline tests only: it learns nothing and knows no cells."""

    dim = 192

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(images, 8).flatten(1)


@torch.no_grad()
def extract(
    model: torch.nn.Module,
    archive_path: Path,
    names: list[str],
    *,
    batch_size: int,
    workers: int,
    device: str,
    dataset: str,
) -> np.ndarray:
    """Features of the named zip members, in that order, as float32 (N, D)."""
    model = model.to(device).eval()
    chunks = []

    def decode(blob: bytes) -> np.ndarray:
        try:
            return preprocess(blob)
        except Exception as exc:
            raise DataError(f"{dataset}: an image cannot be decoded ({type(exc).__name__})") from None

    with zipfile.ZipFile(archive_path) as archive, ThreadPoolExecutor(max(1, workers)) as pool:
        for start in range(0, len(names), batch_size):
            blobs = [read_member(archive, name, dataset) for name in names[start : start + batch_size]]
            batch = torch.from_numpy(np.stack(list(pool.map(decode, blobs)))).to(device)
            chunks.append(model(batch).float().cpu().numpy())
            if (start // batch_size) % 20 == 0:
                print(f"{dataset}: features {min(start + batch_size, len(names))}/{len(names)}", flush=True)
    features = np.concatenate(chunks).astype(np.float32)
    if features.ndim != 2 or len(features) != len(names) or not np.isfinite(features).all():
        raise RuntimeError(f"{dataset}: the features are not a finite (n, d) table")
    return features
