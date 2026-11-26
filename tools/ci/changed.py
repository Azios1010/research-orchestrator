"""CI helper: print, as one line of JSON, what a change touches.

    python tools/ci/changed.py directions --base <ref>   # ["slug", ...] needing an L0 run;
                                                         # all of them if labcore/ changed
    python tools/ci/changed.py configs --base <ref>      # [{"slug": ..., "exp": ...}] to dispatch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from protected_paths import changed_files

from labcore import registry
from labcore.gitutil import GitError, run

SLUG = r"[a-z][a-z0-9-]*"
CONFIG = re.compile(rf"^directions/({SLUG})/configs/([A-Za-z0-9][A-Za-z0-9_-]*)\.yaml$")


def changed_directions(root: Path, base: str, head: str = "HEAD") -> list[str]:
    """Directions that need an L0 run: every one if anything under labcore/ changed, else those
    whose files or registry entry changed. _template is covered by the main tests."""
    root = Path(root)
    files = changed_files(root, base, head)
    if any(name.startswith("labcore/") for name in files):
        return sorted(
            d.name for d in (root / "directions").iterdir()
            if re.fullmatch(SLUG, d.name) and (d / "tests").is_dir()
        )
    slugs = set()
    for name in files:
        match = re.match(rf"^directions/({SLUG})/", name)
        if match:
            slugs.add(match.group(1))
    fork = run(root, "merge-base", base, head).stdout.strip()
    before = run(root, "show", f"{fork}:registry.yaml", check=False)
    old = registry.parse(before.stdout)["directions"] if before.returncode == 0 else {}
    new = registry.load(root / "registry.yaml")["directions"]
    slugs |= {slug for slug, item in new.items() if old.get(slug) != item and re.fullmatch(SLUG, slug)}
    return sorted(s for s in slugs if (root / "directions" / s / "tests").is_dir())


def changed_configs(root: Path, base: str, head: str = "HEAD") -> list[dict]:
    """Experiment configs added or modified (deleted ones are ignored)."""
    found = []
    for name in changed_files(root, base, head):
        match = CONFIG.match(name)
        if match and (Path(root) / name).is_file():
            found.append({"slug": match.group(1), "exp": match.group(2)})
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=("directions", "configs"))
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    function = changed_directions if args.what == "directions" else changed_configs
    try:
        print(json.dumps(function(Path(args.root), args.base, args.head)))
    except (GitError, registry.RegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
