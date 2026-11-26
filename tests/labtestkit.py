"""Helpers shared by the tests (fixtures live in conftest.py)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from labcore import registry
from labcore.io import write_text_atomic
from labcore.kaggle.bundle import code_hash, tree_hash
from labcore.kaggle.client import KaggleError, KernelStatus, OutputResult, PushResult, StatusResult
from labcore.lab import Lab
from labcore.modules import load_module
from labcore.runs.state import StateStore
from labcore.watcher.watcher import Watcher

REPO = Path(__file__).resolve().parents[1]
TOY = "toy-direction"
SECRET_PREFIXES = ("KAGGLE_", "DEEPSEEK_", "TELEGRAM_", "ANTHROPIC_", "CLAUDE_CODE_", "GH_", "GITHUB_", "HF_")


def tool(name: str, root: Path = REPO):
    """Import tools/<name>.py (or tools/ci/<name>.py) as a module."""
    path = root / "tools" / f"{name}.py"
    if not path.is_file():
        path = root / "tools" / "ci" / f"{name}.py"
    return load_module(path)


def git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"git {args}: {done.stderr}"
    return done.stdout.strip()


def commit_all(root: Path, message: str) -> str:
    """Commit inside a throwaway test repository."""
    assert root.resolve() != REPO
    if not (root / ".git").exists():
        git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "HEAD")


def copy_lab(destination: Path, directions: tuple[str, ...] = ()) -> Path:
    """A fresh lab: this repo's code and template, an empty registry, empty state."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("labcore", "tools", "directions/_template", *(f"directions/{d}" for d in directions)):
        shutil.copytree(REPO / name, destination / name, ignore=ignore)
    for name in ("pyproject.toml", ".gitignore", ".gitattributes"):
        shutil.copyfile(REPO / name, destination / name)
    real = registry.load(REPO / "registry.yaml")["directions"]
    registry.save(destination / "registry.yaml", {"directions": {d: real[d] for d in directions}})
    (destination / "results").mkdir()
    (destination / "state").mkdir()
    return destination


def set_gate_approval(root: Path, slug: str, value: str = "APPROVED") -> None:
    path = root / "directions" / slug / "direction.yaml"
    text = re.sub(r"(?m)^(\s+approval:\s*)\w+", rf"\g<1>{value}", path.read_text(encoding="utf-8"), count=1)
    write_text_atomic(path, text)


def mark_l0(root: Path, slug: str) -> None:
    """Record L0_PASS as tools/validate.py --run-l0 would, without running the tests."""
    directory = root / "directions" / slug
    registry.set_status(
        root, slug, "L0_PASS",
        l0={"passed_at": "2026-01-01T00:00:00Z", "code_hash": code_hash(directory),
            "labcore_hash": tree_hash(root / "labcore")},
    )


def approve_and_lock(root: Path, slug: str) -> dict:
    set_gate_approval(root, slug)
    mark_l0(root, slug)
    return tool("lock_prereg").lock(root, slug)


def replace_pipeline(root: Path, body: str, slug: str = TOY) -> None:
    """Swap the direction's entrypoint for `body`, the statements of run(config_path, out_dir)."""
    indented = "\n".join(f"    {line}" for line in body.strip("\n").splitlines())
    write_text_atomic(
        root / "directions" / slug / "src" / "main.py",
        f"def run(config_path, out_dir):\n{indented}\n",
    )


class Clock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 11, 26, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


class FakeKaggle:
    """A backend named 'kaggle' whose status is set by the test. Nothing runs."""

    name = "kaggle"

    def __init__(self, status: KernelStatus = KernelStatus.QUEUED):
        self.current = status
        self.failure_message = None
        self.pushes: list[dict] = []
        self.log = ""
        self.output_error = False

    def push(self, kernel_dir, *, timeout_s=None):
        metadata = json.loads((Path(kernel_dir) / "kernel-metadata.json").read_text())
        self.pushes.append({"metadata": metadata, "timeout_s": timeout_s})
        return PushResult(f"{metadata['id']}/{len(self.pushes)}", len(self.pushes))

    def status(self, ref):
        return StatusResult(self.current, self.current.value, self.failure_message)

    def output(self, ref, dest):
        if self.output_error:
            raise KaggleError("no output available")
        Path(dest).mkdir(parents=True, exist_ok=True)
        return OutputResult(Path(dest), [], self.log)


class RecordingNotifier:
    def __init__(self):
        self.issues: dict[int, dict] = {}
        self.comments: list[tuple[int, str]] = []

    def find_issue(self, title):
        return next((n for n, issue in self.issues.items() if issue["title"] == title), None)

    def create_issue(self, title, body, labels=()):
        number = len(self.issues) + 1
        self.issues[number] = {"title": title, "body": body, "labels": tuple(labels)}
        return number

    def comment(self, number, body):
        self.comments.append((number, body))
        return True


def make_watcher(root: Path, backends: dict, notifier=None, clock=None, **options) -> Watcher:
    options.setdefault("send_telegram", lambda text: False)
    return Watcher(Lab.at(root), backends, notifier or RecordingNotifier(), now=clock or Clock(), **options)


def runs(root: Path) -> list[dict]:
    return StateStore(root / "state").runs()
