"""CI check: no CSV or JSON file under results/ may have phenotype columns (age, sex, NIHSS, mRS ...).

    python tools/ci/results_privacy.py [--root <lab>]

The names that count are listed in labcore/privacy.py. A direction that declares `results_columns`
in direction.yaml is also held to those exact table headers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labcore.config import ConfigError, load_direction
from labcore.privacy import tree_problems


def results_problems(root: Path) -> list[str]:
    """Every problem under results/. Run folders are results/<slug>/<exp>/<run_id>/<output>."""
    results = root / "results"
    problems = tree_problems(results)
    for slug_dir in sorted(p for p in results.glob("*") if p.is_dir()):
        directory = root / "directions" / slug_dir.name
        try:
            columns = load_direction(directory).get("results_columns") if directory.is_dir() else None
        except ConfigError:
            columns = None
        if columns is None:
            continue
        for run_dir in sorted(p for p in slug_dir.glob("*/*") if p.is_dir()):
            for problem in tree_problems(run_dir, columns):
                if "results_columns" in problem:
                    problems.append(f"{run_dir.relative_to(results).as_posix()}/{problem}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    problems = results_problems(Path(args.root))
    for problem in problems:
        print(f"ERROR results/{problem}", file=sys.stderr)
    if problems:
        print(
            "Phenotype data must not be in the repository. Removing the file in a new commit does not "
            "remove it from git history.",
            file=sys.stderr,
        )
        return 1
    print("results/ has no phenotype columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
