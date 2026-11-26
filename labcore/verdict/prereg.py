"""The pre-registration lock: one sha256 over everything that must not move once a run counts.

Covered: prereg.md, the `gates` and `scorecard_map` blocks of direction.yaml, the direction's
checks.py and src/evaluate.py, and labcore.__version__ as the version of the statistics code.

Line endings are normalised and the YAML blocks are hashed as canonical JSON, so the hash is the
same on Windows and Linux and does not change when only YAML comments or layout change. The
version is read from the text of labcore/__init__.py in the checkout being hashed; nothing from
that checkout is imported or executed.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from labcore.io import normalize_newlines, sha256_hex

LOCKED_FILES = ("prereg.md", "checks.py", "src/evaluate.py")
LOCKED_BLOCKS = ("gates", "scorecard_map")
LOCKED_TEXT = "prereg.md, gates, scorecard_map, checks.py, src/evaluate.py or labcore.__version__"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def stats_version(root: Path) -> str:
    """`__version__` as written in <root>/labcore/__init__.py."""
    path = Path(root) / "labcore" / "__init__.py"
    if not path.is_file():
        raise ValueError(f"cannot read the labcore version: {path} not found")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise ValueError(f"cannot read the labcore version: {path}: {exc.msg}") from exc
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError(f"{path} does not assign a literal __version__")


def hash_parts(files: Mapping[str, bytes], direction_yaml_bytes: bytes, version: str) -> str:
    """The lock hash from file contents. `files` maps each name in LOCKED_FILES to its bytes."""
    document = yaml.safe_load(direction_yaml_bytes.decode("utf-8"))
    for block in LOCKED_BLOCKS:
        if not isinstance(document, dict) or block not in document:
            raise ValueError(f"direction.yaml has no {block} block")
    parts = {name: sha256_hex(normalize_newlines(files[name])) for name in LOCKED_FILES}
    parts.update({block: sha256_hex(_canonical(document[block])) for block in LOCKED_BLOCKS})
    parts["labcore.__version__"] = version
    return sha256_hex(_canonical(parts))


def prereg_hash(direction_dir: Path, *, version: str | None = None) -> str:
    """Hash of <root>/directions/<slug>. `version` defaults to the labcore version under <root>."""
    direction_dir = Path(direction_dir).resolve()
    missing = [n for n in (*LOCKED_FILES, "direction.yaml") if not (direction_dir / n).is_file()]
    if missing:
        raise ValueError(f"{direction_dir.name}: missing {', '.join(missing)} (covered by the lock)")
    return hash_parts(
        {name: (direction_dir / name).read_bytes() for name in LOCKED_FILES},
        (direction_dir / "direction.yaml").read_bytes(),
        version if version is not None else stats_version(direction_dir.parents[1]),
    )


def hash_for_entry(direction_dir: Path, item: Mapping[str, Any]) -> str:
    """The hash to compare with the lock recorded in a registry entry.

    While a direction is L1_RUNNING the current labcore version counts, so a version bump breaks
    the lock. A finished direction is compared under the version it was locked with: its files stay
    frozen, but later labcore releases do not invalidate its history.
    """
    lock = item.get("prereg_lock") or {}
    version = None if item.get("status") == "L1_RUNNING" else lock.get("stats_version")
    return prereg_hash(direction_dir, version=version)
