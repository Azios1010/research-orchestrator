"""Split a direction with a GO verdict into its own folder, with history and a pinned labcore.

    python tools/graduate.py <slug> --out ../<slug>

History comes from `git filter-repo` if installed, else `git subtree split`. No remote and no
GitHub repository is created, and nothing is committed in the new folder: the commands to do that
are printed. The lab's registry is marked GRADUATED (commit that through a pull request).
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from labcore import registry
from labcore.config import ConfigError, load_direction
from labcore.gitutil import GitError, head_sha, is_repo_root, run
from labcore.io import iso, read_json, utcnow, write_text_atomic
from labcore.kaggle.bundle import tree_hash
from labcore.verdict.prereg import LOCKED_TEXT, hash_for_entry

METHODS = ("auto", "filter-repo", "subtree")
LAB_ONLY_MODULES = ("tools", "directions")
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")


class GraduationError(RuntimeError):
    pass


def has_filter_repo(root: Path) -> bool:
    return run(root, "filter-repo", "--version", check=False).returncode == 0


def _check_ready(root: Path, slug: str) -> dict:
    item = registry.entry(registry.load(root / "registry.yaml"), slug)
    if item["status"] != "VERDICT" or item.get("verdict") != "GO":
        found = f"{item['status']}" + (f" ({item['verdict']})" if item.get("verdict") else "")
        raise GraduationError(f"{slug} is {found}; only a direction with a GO verdict can graduate")
    directory = root / "directions" / slug
    try:
        current = hash_for_entry(directory, item)
    except (OSError, ValueError) as exc:
        raise GraduationError(f"{slug}: the lock cannot be checked: {exc}") from exc
    if (item.get("prereg_lock") or {}).get("hash") != current:
        raise GraduationError(f"{slug}: {LOCKED_TEXT} differs from the lock")
    if not is_repo_root(root):
        raise GraduationError(f"{root} is not the top of its own git repository")
    paths = (f"directions/{slug}", "labcore", "registry.yaml")
    dirty = run(root, "status", "--porcelain", "--", *paths).stdout.strip()
    if dirty:
        raise GraduationError(f"commit these changes first, or they would be lost:\n{dirty}")
    if run(root, "log", "--oneline", "-1", "--", paths[0], check=False).stdout.strip() == "":
        raise GraduationError(f"directions/{slug} has no committed history")
    return item


def _check_destination(root: Path, out: Path) -> bool:
    """Return True if `out` already existed (empty)."""
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise GraduationError("--out must be outside the lab")
    if not out.exists():
        return False
    if not out.is_dir() or any(out.iterdir()):
        raise GraduationError(f"{out} exists and is not an empty folder")
    return True


def _remove_tree(path: Path) -> None:
    def writable(function, name, _error):
        os.chmod(name, stat.S_IWRITE)  # git marks its object files read-only
        function(name)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=writable)
    else:
        shutil.rmtree(path, onerror=writable)


def _extract_history(root: Path, slug: str, out: Path, method: str) -> None:
    prefix = f"directions/{slug}"
    if method == "filter-repo":
        run(root, "clone", "--quiet", "--no-local", "--no-tags", str(root), str(out))
        run(out, "filter-repo", "--quiet", "--force", "--subdirectory-filter", prefix)
    else:
        branch = f"graduate-tmp-{uuid.uuid4().hex[:8]}"
        run(root, "subtree", "split", "--quiet", f"--prefix={prefix}", "-b", branch)
        try:
            run(
                root, "clone", "--quiet", "--no-local", "--no-tags", "--single-branch",
                "--branch", branch, str(root), str(out),
            )
        finally:
            run(root, "branch", "--quiet", "-D", branch)
    run(out, "branch", "-M", "main")
    for remote in run(out, "remote").stdout.split():
        run(out, "remote", "remove", remote)


def lab_imports(folder: Path) -> list[str]:
    """Imports that only work inside the lab. labcore is fine: it is vendored."""
    found = []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in LAB_ONLY_MODULES:
                    found.append(f"{path.relative_to(folder).as_posix()}:{node.lineno}: imports {name}")
    return found


def _lab_dependencies(root: Path) -> list[str]:
    import tomllib

    with open(root / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["dependencies"]


def _pyproject(root: Path, slug: str) -> str:
    dependencies = "".join(f'    "{item}",\n' for item in _lab_dependencies(root))
    return (
        '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n'
        f'[project]\nname = "{slug}"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        "# Heavy dependencies are listed in requirements.txt.\n"
        f"dependencies = [\n{dependencies}]\n\n"
        '[project.optional-dependencies]\ndev = ["pytest>=8.0", "ruff>=0.6"]\n\n'
        "[tool.setuptools.packages.find]\n"
        'include = ["labcore*"]\n\n'
        "[tool.setuptools.package-data]\n"
        'labcore = ["schemas/*.json", "diagnose/*.md"]\n\n'
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\naddopts = "--import-mode=importlib"\npythonpath = ["."]\n\n'
        '[tool.ruff]\ntarget-version = "py311"\nline-length = 110\n'
    )


def _readme(direction: dict, item: dict, source_sha: str) -> str:
    lock = item["prereg_lock"]
    return f"""# {direction['title']}

Graduated from research-lab (direction `{direction['slug']}`) on {utcnow().date().isoformat()}.

**Question.** {direction['question']}

**Lab verdict: GO**, computed by script from gates locked as `{lock['hash']}` (L1 round {lock['round']}).
A GO verdict means the screening gates passed. It is not a scientific result on its own: read
`VERDICT.md`, including the threats to validity.

## Contents

- `prereg.md`, `direction.yaml`: what was fixed in advance
- `src/`, `configs/`, `checks.py`, `requirements.txt`: the pipeline
- `labcore/`: a pinned copy of the lab library, see `VENDORED.md`
- `evidence/`: scorecards of this direction's lab runs and its registry entry
- Git history of `directions/{direction['slug']}` up to lab commit `{source_sha}`

## Tests

```bash
python -m pip install -e ".[dev]" -r requirements.txt
python -m pytest
```

Data licence: {direction['data']['license']}. No data is stored in this repository.

The lab's dispatcher, watcher workflows and registry are not part of this repository.
"""


def _vendored(root: Path, source_sha: str) -> str:
    from labcore import __version__

    return (
        "# Vendored labcore\n\n"
        f"- version: {__version__}\n"
        f"- lab commit: {source_sha}\n"
        f"- content hash: {tree_hash(root / 'labcore')}\n"
        f"- copied: {iso(utcnow())}\n\n"
        "This is a pinned copy. Changes here do not flow back to the lab.\n"
    )


def _copy_evidence(root: Path, slug: str, item: dict, out: Path) -> None:
    evidence = out / "evidence"
    evidence.mkdir(exist_ok=True)
    results = root / "results" / slug
    for card in sorted(results.glob("*/*/scorecard.json")) if results.is_dir() else []:
        if read_json(card) is not None:
            target = evidence / card.relative_to(results)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(card, target)
    write_text_atomic(
        evidence / "registry_entry.yaml", yaml.safe_dump({slug: item}, sort_keys=False, allow_unicode=True)
    )


def graduate(root: Path, slug: str, out: Path, method: str = "auto") -> dict:
    root, out = Path(root).resolve(), Path(out).resolve()
    if method not in METHODS:
        raise GraduationError(f"unknown method {method!r}")
    item = _check_ready(root, slug)
    existed = _check_destination(root, out)
    direction = load_direction(root / "directions" / slug)
    problems = lab_imports(root / "directions" / slug)
    if problems:
        raise GraduationError("remove imports of lab-only modules first:\n  " + "\n  ".join(problems))
    if method == "auto":
        method = "filter-repo" if has_filter_repo(root) else "subtree"
    elif method == "filter-repo" and not has_filter_repo(root):
        raise GraduationError("git filter-repo is not installed")
    source_sha = head_sha(root)

    try:
        _extract_history(root, slug, out, method)
        shutil.copytree(root / "labcore", out / "labcore", ignore=IGNORED)
        for name, text in (
            ("README.md", _readme(direction, item, source_sha)),
            ("pyproject.toml", _pyproject(root, slug)),
        ):
            # never overwrite a file the direction brought along
            target = out / name if not (out / name).exists() else out / f"GRADUATED-{name}"
            write_text_atomic(target, text)
        write_text_atomic(out / "VENDORED.md", _vendored(root, source_sha))
        for name in (".gitignore", ".gitattributes"):
            if (root / name).is_file() and not (out / name).exists():
                shutil.copyfile(root / name, out / name)
        _copy_evidence(root, slug, item, out)
        registry.set_direction_yaml_status(out, "GRADUATED")
    except Exception:
        if out.exists():  # only what this call created: the folder was absent or empty before
            _remove_tree(out)
            if existed:
                out.mkdir()
        raise

    registry.set_status(root, slug, "GRADUATED", graduated_to=out.name)
    return {"out": out, "method": method, "source_commit": source_sha}


def next_steps(slug: str, out: Path) -> str:
    return f"""
Next steps (yours; nothing below was run):

  1. Review {out}
  2. Test it in a fresh environment:
       cd "{out}"
       python -m pip install -e ".[dev]" -r requirements.txt
       python -m pytest
  3. Commit the vendored library and the new files:
       git -C "{out}" add -A
       git -C "{out}" commit -m "Graduate {slug} from research-lab"
  4. Create the private repository and push:
       gh repo create {slug} --private --source="{out}" --push
  5. In the lab: commit the registry change (status GRADUATED) through a pull request.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("slug")
    parser.add_argument("--out", required=True, help="new folder outside the lab, e.g. ../<slug>")
    parser.add_argument("--method", choices=METHODS, default="auto")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        result = graduate(Path(args.root), args.slug, Path(args.out), args.method)
    except (GraduationError, ConfigError, GitError, registry.RegistryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.slug}: graduated to {result['out']} (history via {result['method']}).")
    print(next_steps(args.slug, result["out"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
