"""Build one self-contained script kernel: labcore + one direction + one experiment config.

The bundle is deterministic (sorted files, zeroed timestamps, LF line endings), so identical
code gives an identical `bundle_hash` on any machine.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import labcore
from labcore.config import load_direction, load_experiment
from labcore.io import normalize_newlines, sha256_hex

log = logging.getLogger("labcore.bundle")

MAX_SCRIPT_BYTES = 800 * 1024
SCRIPT_NAME = "kernel.py"
TEMPLATE = Path(__file__).resolve().parent / "entrypoint_template.py"

EXCLUDED_DIRS = {
    "notebooks", "tests", "results", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints",
    ".dryrun", ".git",
}
# Data and weights never travel with the code.
DATA_SUFFIXES = (
    ".nii", ".gz", ".parquet", ".npz", ".npy", ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".zip", ".tar",
    ".dcm", ".mha", ".nrrd", ".pkl", ".bin", ".onnx", ".safetensors", ".png", ".jpg", ".jpeg", ".pyc",
)
SECRET_NAMES = {".env", "kaggle.json", "access_token", "access_token.txt"}
MANIFEST_MARK = 'MANIFEST_JSON = "{}"  # __LAB_MANIFEST__'
PAYLOAD_MARK = 'PAYLOAD_B64 = ""  # __LAB_PAYLOAD__'


class BundleError(RuntimeError):
    pass


class BundleTooLargeError(BundleError):
    pass


@dataclass
class BundleInfo:
    script_path: Path
    size_bytes: int
    bundle_hash: str
    config_hash: str
    files: list[str]
    warnings: list[str] = field(default_factory=list)


def _walk(root: Path, prefix: str, skip_top: set[str], warnings: list[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts[:-1]) or relative.parts[0] in skip_top:
            continue
        if path.is_symlink() or not path.is_file() or path.suffix == ".pyc":
            continue
        name = f"{prefix}/{relative.as_posix()}"
        if path.suffix.lower() in DATA_SUFFIXES or path.name in SECRET_NAMES:
            warnings.append(f"left out of the bundle (data or secret file): {name}")
            continue
        found[name] = normalize_newlines(path.read_bytes())
    return found


def config_hash(config_path: Path) -> str:
    return sha256_hex(normalize_newlines(Path(config_path).read_bytes()))


def code_hash(direction_dir: Path) -> str:
    """Fingerprint of a direction's code: used for the L0 record and the duplicate-run guard.

    Prose, status, gates and notes are left out, so approving gates does not invalidate L0.
    """
    direction_dir = Path(direction_dir)
    files = _walk(direction_dir, "d", {"configs", "prereg.md", "VERDICT.md", "direction.yaml"}, [])
    document = yaml.safe_load((direction_dir / "direction.yaml").read_text(encoding="utf-8"))
    relevant = {k: v for k, v in document.items() if k not in ("status", "gates", "owner_notes", "title")}
    files["d/direction.yaml"] = json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256_hex(b"".join(name.encode() + b"\0" + files[name] + b"\0" for name in sorted(files)))


def tree_hash(directory: Path) -> str:
    """Fingerprint of every file in a folder that the bundler would ship."""
    files = _walk(Path(directory), "t", set(), [])
    return sha256_hex(b"".join(name.encode() + b"\0" + files[name] + b"\0" for name in sorted(files)))


def _tar_gz(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(files):
                member = tarfile.TarInfo(name)
                member.size = len(files[name])
                member.mtime = 0
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(files[name]))
    return buffer.getvalue()


def build_kernel(
    out_dir: Path,
    *,
    direction_dir: Path,
    config_path: Path,
    run_id: str,
    exp: str,
    commit_sha: str = "unknown",
    labcore_dir: Path | None = None,
    max_bytes: int = MAX_SCRIPT_BYTES,
) -> BundleInfo:
    """Write <out_dir>/kernel.py. Raises BundleTooLargeError above `max_bytes`."""
    direction_dir, config_path = Path(direction_dir), Path(config_path)
    labcore_dir = Path(labcore_dir) if labcore_dir else Path(labcore.__file__).resolve().parent
    direction = load_direction(direction_dir)
    load_experiment(config_path)
    slug = direction["slug"]

    warnings: list[str] = []
    files = _walk(labcore_dir, "labcore", set(), warnings)
    # Only the requested experiment config is shipped; VERDICT.md is not needed to run.
    files.update(_walk(direction_dir, f"directions/{slug}", {"configs", "VERDICT.md"}, warnings))
    config_name = f"directions/{slug}/configs/{config_path.name}"
    files[config_name] = normalize_newlines(config_path.read_bytes())

    module_file = f"directions/{slug}/{direction['entrypoint']['module'].replace('.', '/')}.py"
    if module_file not in files:
        raise BundleError(f"entrypoint module not found in the bundle: {module_file}")

    payload = _tar_gz(files)
    digest = sha256_hex(payload)
    manifest = {
        "run_id": run_id,
        "slug": slug,
        "exp": exp,
        "commit_sha": commit_sha,
        "config_hash": config_hash(config_path),
        "bundle_hash": digest,
        "labcore_version": labcore.__version__,
        "module": direction["entrypoint"]["module"],
        "function": direction["entrypoint"]["function"],
        "config_path": config_name,
        "outputs": direction["outputs"],
    }
    encoded = base64.b64encode(payload).decode("ascii")
    chunks = "".join(f'    "{encoded[i:i + 100]}"\n' for i in range(0, len(encoded), 100))
    script = normalize_newlines(TEMPLATE.read_bytes()).decode("utf-8")
    for mark in (MANIFEST_MARK, PAYLOAD_MARK):
        if script.count(mark) != 1:
            raise BundleError(f"entrypoint template must contain exactly one line: {mark}")
    script = script.replace(MANIFEST_MARK, f"MANIFEST_JSON = {json.dumps(manifest, ensure_ascii=True)!r}")
    script = script.replace(PAYLOAD_MARK, f"PAYLOAD_B64 = (\n{chunks})")
    data = script.encode("utf-8")

    log.info(
        "kernel script size: %d bytes (%.1f%% of the %d byte limit)",
        len(data), 100 * len(data) / max_bytes, max_bytes,
    )
    if len(data) > max_bytes:
        largest = sorted(files, key=lambda n: len(files[n]), reverse=True)[:5]
        raise BundleTooLargeError(
            f"kernel script is {len(data)} bytes; the limit is {max_bytes}. Largest files: "
            + ", ".join(f"{name} ({len(files[name])} B)" for name in largest)
        )
    for warning in warnings:
        log.warning(warning)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / SCRIPT_NAME
    script_path.write_bytes(data)
    return BundleInfo(script_path, len(data), digest, manifest["config_hash"], sorted(files), warnings)
