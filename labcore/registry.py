"""registry.yaml: the authoritative lifecycle state of every direction.

IDEA -> SCAFFOLDED -> L0_PASS -> L1_RUNNING -> VERDICT(GO|REVISE|PARK) -> GRADUATED | ARCHIVED

direction.yaml repeats `status` for readers; the tools keep both in step and
tools/validate.py fails when they disagree.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from labcore.io import read_yaml, write_text_atomic

STATUSES = ("IDEA", "SCAFFOLDED", "L0_PASS", "L1_RUNNING", "VERDICT", "GRADUATED", "ARCHIVED")
VERDICTS = ("GO", "REVISE", "PARK")
HEADER = (
    "# Lifecycle of every direction. Edited by tools/ (through pull requests), not by hand.\n"
    "# IDEA -> SCAFFOLDED -> L0_PASS -> L1_RUNNING -> VERDICT(GO|REVISE|PARK) -> GRADUATED | ARCHIVED\n"
)


class RegistryError(RuntimeError):
    pass


def new_entry(slug: str, title: str, created: str) -> dict[str, Any]:
    return {
        "title": title,
        "status": "SCAFFOLDED",
        "path": f"directions/{slug}",
        "created": created,
        "l0": None,
        "l1_rounds": 0,
        "prereg_lock": None,
        "lock_history": [],
        "verdict": None,
        "graduated_to": None,
    }


def parse(text: str) -> dict[str, Any]:
    document = yaml.safe_load(text) or {}
    document["directions"] = document.get("directions") or {}
    return document


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RegistryError(f"registry not found: {path}")
    return parse(path.read_text(encoding="utf-8"))


def save(path: Path, document: dict[str, Any]) -> None:
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)
    write_text_atomic(path, HEADER + body)


def entry(document: dict[str, Any], slug: str) -> dict[str, Any]:
    try:
        return document["directions"][slug]
    except KeyError:
        raise RegistryError(f"direction {slug!r} is not in registry.yaml") from None


def is_locked(item: dict[str, Any]) -> bool:
    """True once a direction has entered L1: its prereg and gates may no longer change."""
    status = item.get("status")
    if status in ("L1_RUNNING", "VERDICT", "GRADUATED"):
        return True
    return status == "ARCHIVED" and bool(item.get("prereg_lock"))


def set_direction_yaml_status(direction_dir: Path, status: str) -> None:
    """Rewrite only the top-level `status:` line, so comments in direction.yaml survive."""
    path = Path(direction_dir) / "direction.yaml"
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(r"(?m)^status:[^\n#]*", f"status: {status}", text, count=1)
    if count != 1:
        raise RegistryError(f"{path}: no top-level 'status:' line")
    write_text_atomic(path, updated)


def set_status(root: Path, slug: str, status: str, **changes: Any) -> dict[str, Any]:
    """Update registry.yaml and direction.yaml together."""
    if status not in STATUSES:
        raise RegistryError(f"unknown status {status!r}")
    root = Path(root)
    document = load(root / "registry.yaml")
    item = entry(document, slug)
    item["status"] = status
    item.update(changes)
    direction_dir = root / "directions" / slug
    if (direction_dir / "direction.yaml").is_file():
        set_direction_yaml_status(direction_dir, status)
    save(root / "registry.yaml", document)
    return item


def read_status_from_yaml(direction_dir: Path) -> str | None:
    return (read_yaml(Path(direction_dir) / "direction.yaml") or {}).get("status")
