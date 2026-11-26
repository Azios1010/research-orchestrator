"""CI check: a pull request must not touch state/ or results/ (only bot workflows write them).

    python tools/ci/protected_paths.py --base <ref> [--head HEAD]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labcore.gitutil import GitError, run

PROTECTED = ("state/", "results/")


def changed_files(root: Path, base: str, head: str = "HEAD") -> list[str]:
    """Files changed on `head` since it forked from `base`. Renames count on both sides."""
    done = run(root, "diff", "--name-only", "--no-renames", "-z", f"{base}...{head}")
    return [name for name in done.stdout.split("\0") if name]


def violations(root: Path, base: str, head: str = "HEAD") -> list[str]:
    return [name for name in changed_files(root, base, head) if name.lower().startswith(PROTECTED)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        found = violations(Path(args.root), args.base, args.head)
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name in found:
        print(f"ERROR protected path modified: {name}", file=sys.stderr)
    if found:
        print("state/ and results/ are written only by the dispatch and watcher workflows.", file=sys.stderr)
        return 1
    print("protected paths untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
