"""The two image sets: DeepAPL (single cells grouped by patient) and Acevedo (external description).

Both are zip files that the kernel downloads and reads without unpacking. Nothing that identifies a
patient or an image may leave this module in a message: kernel logs and tracebacks can be committed to
results/. So errors here report counts only, and the exceptions that carry a file name are replaced
(`raise ... from None`).
"""

from __future__ import annotations

import hashlib
import http.client
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The twelve cell classes of the probe, in a fixed order: folder names under "Signed slides".
CLASSES = (
    "Blast, no lineage spec",
    "Promyelocyte",
    "Myelocyte",
    "Metamyelocyte",
    "Band neutrophils",
    "Segmented neutrophils",
    "Lymphocyte",
    "Lymphocyte, variant",
    "Monocyte",
    "Eosinophils",
    "Basophil",
    "Erythroblast",
)
SIGNED = "Signed slides"
PATIENT = re.compile(r"Patient_\d+")
IMAGE_SUFFIXES = (".jpg", ".jpeg")

# Acevedo folder -> group of the seven-group comparison. None: not part of it.
ACEVEDO_FOLDERS = {
    "neutrophil": "neutrophil",
    "eosinophil": "eosinophil",
    "basophil": "basophil",
    "lymphocyte": "lymphocyte",
    "monocyte": "monocyte",
    "ig": "immature granulocyte",
    "erythroblast": "erythroblast",
    "platelet": None,
}


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    size: int
    sha256: str


# figshare record 14294675 version 1, "blood smear images_Patient00-105.zip" (md5 2ddc015b...). The
# record's metadata table (diagnosis, age, gender) is a separate file that is never downloaded.
DEEPAPL = Source(
    "deepapl.zip",
    "https://ndownloader.figshare.com/files/27233798",
    539130788,
    "778dce1e2ebb117fcce21665400e6b1f0b6fb59cbcec2c13461bf380f0997f3c",
)
# Mendeley Data 10.17632/snkd93bnjr.1, PBC_dataset_normal_DIB.zip.
ACEVEDO = Source(
    "acevedo.zip",
    "https://data.mendeley.com/public-files/datasets/snkd93bnjr/files/"
    "2fc38728-2ae7-4a62-a857-032af82334c3/file_downloaded",
    281366219,
    "6c50b82d1ab499b0b5a6beda0d0f75e6013df4a7233f116da354351ac10ca4f8",
)
USER_AGENT = "research-lab-kernel"


class DataError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, source: Source) -> None:
    size = Path(path).stat().st_size
    if size != source.size:
        raise DataError(f"{source.name}: {size} bytes, expected {source.size}")
    found = sha256_file(path)
    if found != source.sha256:
        raise DataError(f"{source.name}: sha256 {found}, expected {source.sha256}")


def download(source: Source, folder: Path, attempts: int = 3, wait_s: float = 15.0) -> Path:
    """Fetch one file and check its size and sha256. A wrong file is an error, never used."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target, partial = folder / source.name, folder / f"{source.name}.part"
    if target.is_file():
        verify(target, source)
        return target
    last = "no attempt made"
    for attempt in range(1, attempts + 1):
        try:
            # UNVERIFIED: figshare, Mendeley and Zenodo send a Kaggle kernel the file itself, with no
            # challenge page and no rate limit. Anything else stops the run before any GPU work.
            request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=300) as response:
                status = getattr(response, "status", 200)
                kind = response.headers.get("Content-Type", "")
                if status not in (200, None) or kind.startswith("text/html"):
                    raise DataError(f"the server answered {status} {kind!r} instead of the file")
                with open(partial, "wb") as handle:
                    shutil.copyfileobj(response, handle, 1 << 20)
            verify(partial, source)
            partial.replace(target)
            return target
        except (OSError, http.client.HTTPException, DataError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            print(f"{source.name}: attempt {attempt}/{attempts} failed: {last}", flush=True)
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(wait_s * attempt)
    raise DataError(f"{source.name}: download failed after {attempts} attempts: {last}")


def read_member(archive: zipfile.ZipFile, name: str, dataset: str) -> bytes:
    try:
        return archive.read(name)
    except Exception as exc:
        raise DataError(f"{dataset}: a file in the archive cannot be read ({type(exc).__name__})") from None


@dataclass
class Cells:
    """The kept single-cell images of the twelve classes. `names` are never written anywhere."""

    names: list[str]
    patients: np.ndarray  # patient folder name per cell; never written anywhere
    labels: np.ndarray  # index into CLASSES
    hashes: np.ndarray  # sha256 of the file's bytes
    stats: dict


def _signed_entry(name: str) -> tuple[str, str] | None:
    """(patient folder, class folder) of a signed image, or None if the entry is something else."""
    parts = name.split("/")
    if SIGNED not in parts or "__MACOSX" in parts:
        return None
    at = parts.index(SIGNED)  # there may be a folder between the patient and "Signed slides"
    if len(parts) != at + 3:
        return None
    file_name = parts[-1]
    if file_name.startswith(".") or not file_name.lower().endswith(IMAGE_SUFFIXES):
        return None
    patients = [part for part in parts[:at] if PATIENT.fullmatch(part)]
    if len(patients) != 1:
        raise DataError("DeepAPL: a signed image does not sit under exactly one patient folder")
    return patients[0], parts[at + 1]


def deduplicate(records: list[tuple[str, str, str, str]]) -> tuple[list[tuple[str, str, str, str]], dict]:
    """records: (name, patient, class folder, content hash).

    Images with equal bytes under different classes are all dropped. Equal bytes under one class are
    kept once, the first by name.
    """
    groups: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for record in records:
        groups[record[3]].append(record)
    kept = []
    stats = Counter(
        conflicting_groups=0, conflicting_images=0, repeated_images_dropped=0, repeats_across_patients=0
    )
    for group in groups.values():
        if len({record[2] for record in group}) > 1:
            stats["conflicting_groups"] += 1
            stats["conflicting_images"] += len(group)
            continue
        kept.append(min(group))
        stats["repeated_images_dropped"] += len(group) - 1
        stats["repeats_across_patients"] += len({record[1] for record in group}) > 1
    return sorted(kept), dict(stats)


def index_deepapl(path: Path) -> Cells:
    records = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            entry = _signed_entry(name)
            if entry is None:
                continue
            digest = hashlib.sha256(read_member(archive, name, "DeepAPL")).hexdigest()
            records.append((name, entry[0], entry[1], digest))
    if not records:
        raise DataError(f"DeepAPL: no image under a {SIGNED!r} folder")
    kept, stats = deduplicate(records)
    used = [record for record in kept if record[2] in CLASSES]
    stats.update(n_signed_images=len(records), n_cells=len(used))
    return Cells(
        names=[record[0] for record in used],
        patients=np.array([record[1] for record in used]),
        labels=np.array([CLASSES.index(record[2]) for record in used], dtype=np.int64),
        hashes=np.array([record[3] for record in used]),
        stats=stats,
    )


def class_counts(cells: Cells) -> dict[str, list[int]]:
    """class -> [images, patients]."""
    return {
        name: [int((cells.labels == k).sum()), len(set(cells.patients[cells.labels == k].tolist()))]
        for k, name in enumerate(CLASSES)
    }


def index_acevedo(path: Path) -> tuple[list[str], list[str]]:
    """(member names, folder of each) for every image, platelets included."""
    names, folders = [], []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            parts = name.split("/")
            file_name = parts[-1]
            if "__MACOSX" in parts or file_name.startswith("."):
                continue
            if len(parts) < 2 or not file_name.lower().endswith(IMAGE_SUFFIXES):
                continue
            if parts[-2] not in ACEVEDO_FOLDERS:
                raise DataError("Acevedo: an image sits in a folder that is not one of the eight groups")
            names.append(name)
            folders.append(parts[-2])
    if not names:
        raise DataError("Acevedo: no image in the archive")
    return names, folders


def count_problems(expected: dict, found: dict, what: str) -> list[str]:
    """Differences between two {name: count or [counts]} tables, as short messages."""
    return [
        f"{what} {name!r}: found {found.get(name)}, expected {expected.get(name)}"
        for name in sorted(set(expected) | set(found))
        if found.get(name) != expected.get(name)
    ]
