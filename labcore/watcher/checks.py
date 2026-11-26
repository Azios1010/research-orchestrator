"""Generic sanity checks on a downloaded run, and the hook for direction-specific ones."""

from __future__ import annotations

import shutil
from pathlib import Path

from labcore.io import read_json
from labcore.kaggle.bundle import DATA_SUFFIXES
from labcore.modules import load_module
from labcore.privacy import file_problems

ALWAYS_KEPT = ("status.json", "env.json")
MAX_RESULT_BYTES = 5 * 1024 * 1024


def copy_results(
    output_dir: Path, result_dir: Path, required: list[str], columns: dict | None = None
) -> list[str]:
    """Copy only the declared outputs into the repo. Data files, large files and files with
    phenotype columns are refused.

    Anything else the kernel wrote (figures, caches) stays on Kaggle and is never committed.
    """
    problems = []
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    root = Path(output_dir).resolve()
    for name in dict.fromkeys((*ALWAYS_KEPT, *required)):
        source = (root / name).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            continue
        if source.suffix.lower() in DATA_SUFFIXES:
            problems.append(f"refusing to store data-like output: {name}")
        elif source.stat().st_size > MAX_RESULT_BYTES:
            problems.append(f"output is larger than {MAX_RESULT_BYTES} bytes and was not stored: {name}")
        elif found := file_problems(source, name, columns):
            problems += [f"not stored: {problem}" for problem in found]
        else:
            target = result_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return problems


def generic_checks(result_dir: Path, required: list[str], columns: dict | None = None) -> list[str]:
    result_dir = Path(result_dir)
    problems = []
    try:
        status = read_json(result_dir / "status.json")
    except ValueError:
        status = None
        problems.append("status.json is not valid JSON")
    if status is None:
        problems.append("status.json is missing")
    elif status.get("ok") is not True:
        problems.append(f"status.json is not ok (stage {status.get('stage')!r})")
    for name in required:
        path = result_dir / name
        if not path.is_file():
            problems.append(f"required output is missing: {name}")
        elif path.stat().st_size == 0:
            problems.append(f"required output is empty: {name}")
    # Also in L0, so that a forbidden column is found before any GPU time is spent.
    for name in dict.fromkeys((*ALWAYS_KEPT, *required)):
        if (result_dir / name).is_file():
            problems += file_problems(result_dir / name, name, columns)
    return problems


def direction_checks(checks_path: Path, result_dir: Path) -> list[str]:
    """Run directions/<slug>/checks.py::check. A crashing check counts as a failed check."""
    checks_path = Path(checks_path)
    if not checks_path.is_file():
        return []
    try:
        return [str(problem) for problem in load_module(checks_path).check(Path(result_dir))]
    except Exception as exc:  # reported, not raised: the watcher must keep going
        return [f"checks.py crashed: {type(exc).__name__}: {exc}"]
