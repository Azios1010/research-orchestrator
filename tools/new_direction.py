"""Create a direction from directions/_template and register it as SCAFFOLDED.

    python tools/new_direction.py my-slug --title "One line question"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labcore import registry
from labcore.io import utcnow, write_text_atomic

SLUG = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
MAX_SLUG = 30
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".json"}


class ScaffoldError(RuntimeError):
    pass


def new_direction(root: Path, slug: str, title: str) -> Path:
    root = Path(root)
    if not SLUG.match(slug) or len(slug) > MAX_SLUG:
        raise ScaffoldError(f"slug must be lowercase words joined by '-', at most {MAX_SLUG} characters")
    title = " ".join(title.split())
    if not title:
        raise ScaffoldError("title must not be empty")
    document = registry.load(root / "registry.yaml")
    existing = document["directions"].get(slug)
    if existing and existing.get("status") != "IDEA":
        raise ScaffoldError(f"{slug} is already registered with status {existing['status']}")
    template, target = root / "directions" / "_template", root / "directions" / slug
    if target.exists():
        raise ScaffoldError(f"{target} already exists")
    shutil.copytree(template, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    created = utcnow().date().isoformat()
    replacements = {
        "__SLUG__": slug,
        "__TITLE_JSON__": json.dumps(title, ensure_ascii=False),  # a JSON string is valid YAML
        "__TITLE__": title,
        "__DATE__": created,
    }
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            for mark, value in replacements.items():
                text = text.replace(mark, value)
            write_text_atomic(path, text)
    document["directions"][slug] = registry.new_entry(slug, title, created)
    registry.save(root / "registry.yaml", document)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("slug")
    parser.add_argument("--title", required=True)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        target = new_direction(Path(args.root), args.slug, args.title)
    except (ScaffoldError, registry.RegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"created {target} (SCAFFOLDED)")
    print("next: edit direction.yaml and prereg.md, replace the fake data and pipeline, then")
    print(f"      python tools/validate.py --run-l0 {args.slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
