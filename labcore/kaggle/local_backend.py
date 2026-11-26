"""Run a bundled kernel on this machine, on CPU, behind the same interface as KaggleClient.

Used by the tests and by `--dry-run`. The kernel runs to completion inside `push`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from labcore.io import read_json, write_json
from labcore.kaggle.client import KaggleError, KernelStatus, OutputResult, PushResult, StatusResult

SECRET_PREFIXES = ("KAGGLE_", "DEEPSEEK_", "TELEGRAM_", "ANTHROPIC_", "CLAUDE_CODE_", "GH_", "GITHUB_", "HF_")


class LocalBackend:
    name = "local"

    def __init__(self, work_root: Path, python: str = sys.executable, timeout_s: int = 3600):
        self.work_root = Path(work_root).resolve()
        self.python = python
        self.timeout_s = timeout_s

    def _run_dir(self, ref: str) -> Path:
        parts = ref.split("/")
        if len(parts) != 3:
            raise KaggleError(f"local kernel refs look like owner/slug/version, got {ref!r}")
        return self.work_root / parts[1] / f"v{int(parts[2])}"

    def push(self, kernel_dir: Path, *, timeout_s: int | None = None) -> PushResult:
        kernel_dir = Path(kernel_dir)
        metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        slug = metadata["id"].split("/")[1]
        base = self.work_root / slug
        base.mkdir(parents=True, exist_ok=True)
        version = 1 + max((int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit()), default=0)
        run_dir = base / f"v{version}"
        working = run_dir / "working"
        working.mkdir(parents=True)
        script = run_dir / metadata["code_file"]
        shutil.copyfile(kernel_dir / metadata["code_file"], script)

        # The kernel gets no credentials and imports only the bundled code.
        env = {k: v for k, v in os.environ.items() if not k.startswith(SECRET_PREFIXES)}
        env.pop("PYTHONPATH", None)
        env.update(LAB_KAGGLE_WORKING=str(working), LAB_SKIP_PIP="1", PYTHONIOENCODING="utf-8")
        env.setdefault("MPLBACKEND", "Agg")
        log_path = run_dir / "kernel.log"
        failure = None
        with open(log_path, "w", encoding="utf-8") as log_file:
            try:
                done = subprocess.run(
                    [self.python, "-u", str(script)], cwd=run_dir, env=env, stdout=log_file,
                    stderr=subprocess.STDOUT, timeout=self.timeout_s, check=False,
                )
                status = KernelStatus.COMPLETE if done.returncode == 0 else KernelStatus.ERROR
            except subprocess.TimeoutExpired:
                status, failure = KernelStatus.ERROR, f"local run exceeded {self.timeout_s}s"
        write_json(run_dir / "state.json", {"status": status.value, "failure_message": failure})
        return PushResult(f"{metadata['id']}/{version}", version, f"local run finished: {status.value}")

    def status(self, ref: str) -> StatusResult:
        state = read_json(self._run_dir(ref) / "state.json")
        if state is None:
            return StatusResult(KernelStatus.UNKNOWN, "no local run")
        return StatusResult(KernelStatus(state["status"]), state["status"], state.get("failure_message"))

    def output(self, ref: str, dest: Path) -> OutputResult:
        run_dir = self._run_dir(ref)
        if not run_dir.is_dir():
            raise KaggleError(f"no local run for {ref}")
        dest = Path(dest)
        shutil.copytree(run_dir / "working", dest, dirs_exist_ok=True)
        log_path = run_dir / "kernel.log"
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        return OutputResult(dest, [p for p in dest.rglob("*") if p.is_file()], text)
