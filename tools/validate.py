"""Validate the registry, every direction and the state files; optionally run a direction's L0.

    python tools/validate.py                      # everything
    python tools/validate.py isles24-ctp          # one direction
    python tools/validate.py --run-l0 isles24-ctp # run its L0 tests and record L0_PASS
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labcore import registry
from labcore.config import ConfigError, load_direction, load_experiment
from labcore.io import iso, utcnow
from labcore.kaggle.bundle import code_hash, tree_hash
from labcore.kaggle.metadata import kernel_slug
from labcore.runs.state import StateError, StateStore
from labcore.verdict.prereg import LOCKED_TEXT, hash_for_entry

REQUIRED_FILES = (
    "prereg.md", "requirements.txt", "checks.py", "src/evaluate.py", "VERDICT.md",
    "tests/fake_data.py", "tests/test_l0_smoke.py",
)


def direction_errors(root: Path, slug: str, item: dict) -> list[str]:
    status = item.get("status")
    if status not in registry.STATUSES:
        return [f"unknown status {status!r}"]
    directory = root / "directions" / slug
    if not directory.is_dir():
        if status in ("IDEA", "GRADUATED", "ARCHIVED"):
            return []
        return [f"folder directions/{slug} is missing"]
    try:
        direction = load_direction(directory)
    except ConfigError as exc:
        return [str(exc)]
    errors = []
    if direction["slug"] != slug:
        errors.append(f"direction.yaml slug is {direction['slug']!r}")
    if direction["status"] != status:
        errors.append(f"direction.yaml status {direction['status']} differs from registry status {status}")
    errors += [f"missing {name}" for name in REQUIRED_FILES if not (directory / name).is_file()]
    module = directory / f"{direction['entrypoint']['module'].replace('.', '/')}.py"
    if not module.is_file():
        errors.append(f"entrypoint module not found: {module.relative_to(directory).as_posix()}")
    configs = sorted((directory / "configs").glob("*.yaml"))
    if not configs:
        errors.append("no experiment config in configs/")
    for config in configs:
        try:
            estimate = load_experiment(config)["compute"]["est_gpu_hours"]
            kernel_slug(slug, config.stem)
        except (ConfigError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if estimate > direction["compute"]["l1_gpu_hours_cap"]:
            errors.append(f"configs/{config.name}: est_gpu_hours {estimate} exceeds l1_gpu_hours_cap")
    if status == "VERDICT" and item.get("verdict") not in registry.VERDICTS:
        errors.append("status is VERDICT but the registry has no GO/REVISE/PARK verdict")
    if registry.is_locked(item):
        lock = item.get("prereg_lock") or {}
        try:
            if lock.get("hash") != hash_for_entry(directory, item):
                errors.append(f"{LOCKED_TEXT} differs from the lock in registry.yaml")
        except (OSError, ValueError) as exc:
            errors.append(f"the lock cannot be checked: {exc}")
    return errors


def validate_lab(root: Path, slugs: list[str] | None = None) -> list[str]:
    root = Path(root)
    try:
        directions = registry.load(root / "registry.yaml")["directions"]
    except (registry.RegistryError, ValueError) as exc:
        return [str(exc)]
    errors = []
    for slug in slugs or sorted(directions):
        if slug not in directions:
            errors.append(f"{slug}: not in registry.yaml")
            continue
        errors += [f"{slug}: {error}" for error in direction_errors(root, slug, directions[slug])]
    if slugs:
        return errors
    on_disk = root / "directions"
    for directory in sorted(on_disk.iterdir()) if on_disk.is_dir() else []:
        hidden = directory.name.startswith(("_", "."))
        if directory.is_dir() and not hidden and directory.name not in directions:
            errors.append(f"directions/{directory.name} is not registered in registry.yaml")
    store = StateStore(root / "state")
    for read in (store.runs, store.queue, store.budget):
        try:
            read()
        except (StateError, ValueError) as exc:
            errors.append(str(exc))
    return errors


def run_l0(root: Path, slug: str, record: bool = True) -> tuple[bool, str]:
    """Run directions/<slug>/tests. Skipped tests do not count as a pass."""
    root = Path(root).resolve()
    item = registry.entry(registry.load(root / "registry.yaml"), slug)
    directory = root / "directions" / slug
    with tempfile.TemporaryDirectory(prefix="lab_l0_") as temporary:
        report = Path(temporary) / "junit.xml"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
        command = [
            sys.executable, "-m", "pytest", str(directory / "tests"), "-q", f"--junitxml={report}",
            "--rootdir", str(root), "-p", "no:cacheprovider",
        ]
        done = subprocess.run(command, cwd=root, env=env, check=False)
        totals = {"tests": 0, "skipped": 0}
        if report.is_file():
            for suite in ElementTree.parse(report).getroot().iter("testsuite"):
                for key in totals:
                    totals[key] += int(suite.get(key, 0))
    passed = done.returncode == 0 and totals["tests"] > 0 and totals["skipped"] == 0
    if done.returncode != 0:
        detail = f"pytest exit code {done.returncode}"
    elif totals["skipped"]:
        detail = f"{totals['skipped']} test(s) skipped: install the direction's requirements"
    else:
        detail = f"{totals['tests']} test(s) passed"
    message = f"L0 {'PASS' if passed else 'FAIL'} for {slug}: {detail}"
    if record and item["status"] in ("SCAFFOLDED", "L0_PASS"):
        if passed:
            registry.set_status(
                root, slug, "L0_PASS",
                l0={
                    "passed_at": iso(utcnow()),
                    "code_hash": code_hash(directory),
                    "labcore_hash": tree_hash(root / "labcore"),
                },
            )
            message += " (registry: L0_PASS)"
        elif item["status"] == "L0_PASS":
            registry.set_status(root, slug, "SCAFFOLDED", l0=None)
            message += " (registry: back to SCAFFOLDED)"
    elif record:
        message += f" (registry status stays {item['status']})"
    return passed, message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("slugs", nargs="*")
    parser.add_argument("--run-l0", action="store_true", help="run the L0 tests of the given directions")
    parser.add_argument("--no-record", action="store_true", help="with --run-l0: do not edit the registry")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    root = Path(args.root)
    errors = validate_lab(root, args.slugs or None)
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    if errors:
        return 1
    print("validation passed")
    if not args.run_l0:
        return 0
    if not args.slugs:
        parser.error("--run-l0 needs at least one slug")
    ok = True
    for slug in args.slugs:
        passed, message = run_l0(root, slug, record=not args.no_record)
        print(message)
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
