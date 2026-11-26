"""CI check: locked pre-registrations must not change.

    python tools/ci/prereg_lock.py [--base <ref>]

Without --base: every direction at status >= L1_RUNNING must still match its recorded hash over
prereg.md, gates, scorecard_map, checks.py, src/evaluate.py and labcore.__version__. A finished
direction is compared under the labcore version it was locked with.
With --base: the lock recorded on the base branch must also be intact, so a pull request cannot
edit the gates and re-lock them, or quietly move the status back; and a direction that is
L1_RUNNING on the base is compared under the current labcore version.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labcore import registry
from labcore.gitutil import GitError, run
from labcore.verdict.prereg import LOCKED_TEXT, hash_for_entry


def current_errors(root: Path, directions: dict, running: frozenset[str] = frozenset()) -> list[str]:
    """`running` names directions that are L1_RUNNING on the base branch."""
    errors = []
    for slug, item in directions.items():
        if not registry.is_locked(item):
            continue
        lock = item.get("prereg_lock") or {}
        directory = root / "directions" / slug
        if not lock.get("hash"):
            errors.append(f"{slug}: status {item['status']} but no prereg lock")
        elif not directory.is_dir():
            if item["status"] not in ("GRADUATED", "ARCHIVED"):
                errors.append(f"{slug}: locked direction folder is missing")
        else:
            try:
                compared = {**item, "status": "L1_RUNNING"} if slug in running else item
                actual = hash_for_entry(directory, compared)
            except (OSError, ValueError) as exc:
                errors.append(f"{slug}: cannot hash the locked files: {exc}")
                continue
            if actual != lock["hash"]:
                errors.append(
                    f"{slug}: {LOCKED_TEXT} changed after the lock "
                    f"(locked {lock['hash'][:19]}, now {actual[:19]})"
                )
    return errors


def history_errors(base: dict, head: dict) -> list[str]:
    """Compare locked entries of the base branch with the pull request."""
    errors = []
    for slug, old in base.items():
        old_history = old.get("lock_history") or []
        new = head.get(slug)
        if new is None:
            if registry.is_locked(old) or old_history:
                errors.append(f"{slug}: a direction with a prereg lock was removed from the registry")
            continue
        history = new.get("lock_history") or []
        if history[: len(old_history)] != old_history:
            errors.append(f"{slug}: lock_history was rewritten")
        if new.get("l1_rounds", 0) < old.get("l1_rounds", 0):
            errors.append(f"{slug}: l1_rounds was reduced")
        if not registry.is_locked(old):
            continue
        old_lock = old.get("prereg_lock") or {}
        opened_new_round = (
            old.get("status") == "VERDICT"
            and old.get("verdict") == "REVISE"
            and any(h.get("hash") == old_lock.get("hash") and h.get("round") == old_lock.get("round")
                    for h in history[len(old_history):])
        )
        if opened_new_round:
            continue
        if not registry.is_locked(new):
            errors.append(f"{slug}: status moved from {old['status']} back to {new.get('status')}")
        if new.get("prereg_lock") != old_lock:
            errors.append(f"{slug}: the prereg lock recorded on the base branch was changed or removed")
    return errors


def check(root: Path, base: str | None = None) -> list[str]:
    root = Path(root)
    head = registry.load(root / "registry.yaml")["directions"]
    old: dict = {}
    if base:
        if run(root, "rev-parse", "--verify", "-q", f"{base}^{{commit}}", check=False).returncode != 0:
            raise GitError(f"base ref {base!r} not found (is the checkout shallow?)")
        shown = run(root, "show", f"{base}:registry.yaml", check=False)
        if shown.returncode == 0:
            old = registry.parse(shown.stdout)["directions"]
    running = frozenset(
        slug for slug, item in old.items()
        if item.get("status") == "L1_RUNNING"
        and (head.get(slug) or {}).get("prereg_lock") == item.get("prereg_lock")
    )
    return current_errors(root, head, running) + history_errors(old, head)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        errors = check(Path(args.root), args.base)
    except (GitError, registry.RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    if errors:
        return 1
    print("prereg locks intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
