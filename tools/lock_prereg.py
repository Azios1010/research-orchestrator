"""Hash-lock a direction and move it into L1 (or open a new round).

The lock covers prereg.md, gates, scorecard_map, checks.py, every file under src/ and
labcore.__version__ (hash scheme 2).

    python tools/lock_prereg.py <slug>              # L0_PASS -> L1_RUNNING
    python tools/lock_prereg.py <slug> --new-round  # after a REVISE verdict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labcore import registry
from labcore.config import ConfigError, load_direction
from labcore.io import iso, utcnow, write_text_atomic
from labcore.kaggle.bundle import code_hash, tree_hash
from labcore.verdict.prereg import LOCKED_TEXT, SCHEME, prereg_hash, stats_version


class LockError(RuntimeError):
    pass


def lock(root: Path, slug: str) -> dict:
    root = Path(root)
    item = registry.entry(registry.load(root / "registry.yaml"), slug)
    directory = root / "directions" / slug
    if item["status"] != "L0_PASS":
        raise LockError(
            f"{slug} is {item['status']}; only an L0_PASS direction can enter L1. "
            f"Run: python tools/validate.py --run-l0 {slug}"
        )
    direction = load_direction(directory)
    if direction["gates"]["approval"] != "APPROVED":
        raise LockError(
            f"{slug}: gates are marked {direction['gates']['approval']}. The owner must review them and "
            "set gates.approval to APPROVED in direction.yaml before they can be locked."
        )
    recorded = item.get("l0") or {}
    current = {"code_hash": code_hash(directory), "labcore_hash": tree_hash(root / "labcore")}
    if any(recorded.get(key) != value for key, value in current.items()):
        raise LockError(f"{slug}: code changed since L0 passed; run L0 again")
    maximum = direction["compute"]["max_l1_rounds"]
    if item.get("l1_rounds", 0) >= maximum:
        raise LockError(f"{slug}: all {maximum} L1 rounds are used")
    if not (directory / "prereg.md").read_text(encoding="utf-8").strip():
        raise LockError(f"{slug}: prereg.md is empty")
    round_number = item.get("l1_rounds", 0) + 1
    try:
        version = stats_version(root)
        digest = prereg_hash(directory, version=version, scheme=SCHEME)
    except (OSError, ValueError) as exc:
        raise LockError(str(exc)) from exc
    new_lock = {
        "hash": digest, "locked_at": iso(utcnow()), "round": round_number, "stats_version": version,
        "hash_scheme": SCHEME,
    }
    registry.set_status(
        root, slug, "L1_RUNNING", l1_rounds=round_number, prereg_lock=new_lock, verdict=None
    )
    return new_lock


def new_round(root: Path, slug: str) -> None:
    """Reopen a REVISE direction. The old lock is kept in lock_history; gates return to DRAFT."""
    root = Path(root)
    item = registry.entry(registry.load(root / "registry.yaml"), slug)
    directory = root / "directions" / slug
    if item["status"] != "VERDICT" or item.get("verdict") != "REVISE":
        raise LockError(f"{slug}: a new round is only possible after a REVISE verdict")
    maximum = load_direction(directory)["compute"]["max_l1_rounds"]
    if item.get("l1_rounds", 0) >= maximum:
        raise LockError(f"{slug}: all {maximum} L1 rounds are used; PARK it or start a new direction")
    history = [*item.get("lock_history", []), {**item["prereg_lock"], "verdict": item["verdict"]}]
    path = directory / "direction.yaml"
    text = path.read_text(encoding="utf-8")
    write_text_atomic(path, re.sub(r"(?m)^(\s+approval:\s*)APPROVED", r"\g<1>DRAFT", text, count=1))
    registry.set_status(
        root, slug, "SCAFFOLDED", lock_history=history, prereg_lock=None, verdict=None, l0=None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("slug")
    parser.add_argument("--new-round", action="store_true")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        if args.new_round:
            new_round(Path(args.root), args.slug)
            print(f"{args.slug}: new round open (SCAFFOLDED, gates DRAFT).")
            print("Edit, re-run L0, get the gates approved, then lock.")
        else:
            locked = lock(Path(args.root), args.slug)
            print(f"{args.slug}: L1_RUNNING, round {locked['round']}, {locked['hash']}")
            print("Commit this through a pull request. Now frozen:")
            print(f"  {LOCKED_TEXT}")
    except (LockError, ConfigError, registry.RegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
