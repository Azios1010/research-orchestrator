"""CI check: no CSV or JSON file under results/ may have phenotype columns (age, sex, NIHSS, mRS ...).

    python tools/ci/results_privacy.py [--root <lab>]

The names that count are listed in labcore/privacy.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labcore.privacy import tree_problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    problems = tree_problems(Path(args.root) / "results")
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
