"""The few git operations the lab needs, always pinned to the lab's own repository.

`git -C <root>` would silently fall through to an enclosing repository, so every helper first
checks that <root> itself is the top of a work tree.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path


class GitError(RuntimeError):
    pass


def run(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    done = subprocess.run(
        ["git", "-c", "core.longpaths=true", "-C", str(root), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if check and done.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {done.stderr.strip() or done.stdout.strip()}")
    return done


def is_repo_root(root: Path) -> bool:
    done = run(root, "rev-parse", "--show-toplevel", check=False)
    if done.returncode != 0:
        return False
    try:
        return os.path.samefile(done.stdout.strip(), root)
    except OSError:
        return False


def head_sha(root: Path, paths: tuple[str, ...] = ()) -> str:
    """HEAD commit, with '-dirty' if `paths` have uncommitted changes; 'unknown' without a commit."""
    if not is_repo_root(root):
        return "unknown"
    head = run(root, "rev-parse", "--verify", "-q", "HEAD", check=False)
    if head.returncode != 0:
        return "unknown"
    sha = head.stdout.strip()
    if paths and run(root, "status", "--porcelain", "--", *paths).stdout.strip():
        sha += "-dirty"
    return sha


def has_commit(root: Path, sha: str) -> bool:
    return run(root, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0


def extract_commit(root: Path, sha: str, paths: list[str], dest: Path) -> bool:
    """Unpack `paths` as they were at `sha` into `dest`. Fetches the commit if needed."""
    if not is_repo_root(root) or not all(c in "0123456789abcdef" for c in sha) or len(sha) < 7:
        return False
    if not has_commit(root, sha):
        # UNVERIFIED: GitHub serves a commit by its sha, e.g. the head of a merged or closed PR.
        run(root, "fetch", "--quiet", "origin", sha, check=False)
        if not has_commit(root, sha):
            return False
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    present = [p for p in paths if run(root, "cat-file", "-e", f"{sha}:{p}", check=False).returncode == 0]
    if len(present) != len(paths):
        return False
    archive = dest / "_snapshot.tar"
    run(root, "archive", "--format=tar", f"--output={archive}", sha, "--", *paths)
    with tarfile.open(archive) as handle:
        handle.extractall(dest, filter="data")
    archive.unlink()
    return True
