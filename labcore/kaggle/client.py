"""Thin subprocess wrapper around the `kaggle` CLI, plus the interface every backend implements.

Behaviour below was read from the source of kaggle 2.2.5:

* `kernels push` exits 0 even when the push fails; failure is the text "Kernel push error".
* `kernels status` prints: <ref> has status "<value>" and optionally: Failure message: "<text>"
* `kernels output` writes the output files and <kernel-slug>.log (a JSON list of log events).
* A ref may carry a version: owner/slug/3.
* Auth: KAGGLE_API_TOKEN (access token) is tried first, then KAGGLE_USERNAME + KAGGLE_KEY.

None of this has been exercised against the real service. See SETUP.md, first-run checklist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class KernelStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


# UNVERIFIED: status names and the `has status "..."` / `Failure message` text of `kernels status`.
_STATUS_BY_NAME = {
    "queued": KernelStatus.QUEUED,
    "newscript": KernelStatus.QUEUED,
    "running": KernelStatus.RUNNING,
    "cancelrequested": KernelStatus.RUNNING,  # still shutting down
    "complete": KernelStatus.COMPLETE,
    "error": KernelStatus.ERROR,
    "cancelacknowledged": KernelStatus.CANCELLED,
    "cancelled": KernelStatus.CANCELLED,
}


@dataclass
class PushResult:
    ref: str  # owner/slug or owner/slug/version
    version: int | None
    message: str = ""


@dataclass
class StatusResult:
    status: KernelStatus
    raw: str = ""
    failure_message: str | None = None


@dataclass
class OutputResult:
    dest: Path
    files: list[Path] = field(default_factory=list)
    log_text: str = ""


class KaggleError(RuntimeError):
    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


class Backend(Protocol):
    name: str

    def push(self, kernel_dir: Path, *, timeout_s: int | None = None) -> PushResult: ...

    def status(self, ref: str) -> StatusResult: ...

    def output(self, ref: str, dest: Path) -> OutputResult: ...


def parse_status(text: str) -> StatusResult:
    match = re.search(r'has status "([^"]+)"', text)
    if not match:
        return StatusResult(KernelStatus.UNKNOWN, text.strip())
    # e.g. "KernelWorkerStatus.CANCEL_ACKNOWLEDGED", "complete" or "cancelAcknowledged"
    name = match.group(1).split(".")[-1].replace("_", "").lower()
    failure = re.search(r'Failure message: "(.*)"', text, re.DOTALL)
    return StatusResult(
        _STATUS_BY_NAME.get(name, KernelStatus.UNKNOWN),
        match.group(1),
        failure.group(1).strip() if failure else None,
    )


def parse_push(text: str) -> int | None:
    """Return the new kernel version, or raise KaggleError if the push did not happen."""
    # UNVERIFIED: the success and error texts printed by `kernels push` (it exits 0 either way).
    error = re.search(r"Kernel push error:\s*(.*)", text)
    if error:
        raise KaggleError(f"kernel push failed: {error.group(1).strip()}")
    if "not valid dataset sources" in text:
        raise KaggleError(f"kernel was pushed without some dataset sources: {text.strip()}")
    if "successfully pushed" not in text:
        raise KaggleError(f"kernel push gave no success message: {text.strip()[:500]}")
    version = re.search(r"Kernel version (\d+) successfully pushed", text)
    return int(version.group(1)) if version else None


def log_to_text(raw: str) -> str:
    """Turn Kaggle's JSON log (a list of {stream_name, time, data}) into plain text."""
    try:
        events = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw or ""
    if not isinstance(events, list):
        return raw
    return "".join(str(e.get("data", "")) for e in events if isinstance(e, dict))


# Errors that mean the request never got a proper answer, so asking again is safe.
_TRANSIENT = re.compile(
    r"connection ?(error|reset|refused|aborted)|failed to establish|name resolution|getaddrinfo|"
    r"max retries exceeded|\b(429|500|502|503|504)\b.*(error|server|gateway|unavailable|requests)|"
    r"too many requests|service unavailable|bad gateway|gateway time-?out|timed out",
    re.IGNORECASE,
)
# A push is only repeated when it clearly never reached the server: repeating one that did
# arrive would start a second GPU session.
_PUSH_SAFE_TO_REPEAT = re.compile(
    r"failed to establish|name resolution|getaddrinfo|connection refused", re.IGNORECASE
)


class KaggleClient:
    name = "kaggle"

    def __init__(
        self,
        executable: str = "kaggle",
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.executable = executable
        self.runner = runner
        self.retries = retries
        self.sleep = sleep

    def _environment(self) -> dict[str, str]:
        # An unset GitHub secret arrives as an empty string; the CLI would treat it as a value.
        env = {k: v for k, v in os.environ.items() if not (k.startswith("KAGGLE_") and not v)}
        env["PYTHONIOENCODING"] = "utf-8"
        # UNVERIFIED: the CLI reads KAGGLE_API_TOKEN, else KAGGLE_USERNAME + KAGGLE_KEY, from here.
        return env

    def _run(self, args: list[str], *, timeout: int, repeat_if: re.Pattern = _TRANSIENT) -> str:
        command = [shutil.which(self.executable) or self.executable, *args]
        last = ""
        for attempt in range(1, self.retries + 1):
            try:
                done = self.runner(
                    command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=timeout, env=self._environment(), check=False,
                )
            except FileNotFoundError as exc:
                raise KaggleError(f"kaggle CLI not found: {exc}") from exc
            except subprocess.TimeoutExpired:
                last = f"`kaggle {' '.join(args[:2])}` timed out after {timeout}s"
                retry = repeat_if is _TRANSIENT
            else:
                output = f"{done.stdout or ''}\n{done.stderr or ''}"
                push_error = "Kernel push error" in output
                if done.returncode == 0 and not (push_error and repeat_if.search(output)):
                    return output
                last = output.strip()[-2000:]
                retry = bool(repeat_if.search(output))
            if not retry or attempt == self.retries:
                raise KaggleError(
                    f"`kaggle {' '.join(args[:2])}` failed: {last}", transient=bool(_TRANSIENT.search(last))
                )
            self.sleep(2.0**attempt)
        raise KaggleError(last)

    def push(self, kernel_dir: Path, *, timeout_s: int | None = None) -> PushResult:
        metadata = json.loads((Path(kernel_dir) / "kernel-metadata.json").read_text(encoding="utf-8"))
        args = ["kernels", "push", "-p", str(kernel_dir)]
        # UNVERIFIED: that -t stops the Kaggle session after this many seconds.
        if timeout_s:
            args += ["-t", str(int(timeout_s))]
        output = self._run(args, timeout=300, repeat_if=_PUSH_SAFE_TO_REPEAT)
        version = parse_push(output)
        ref = f"{metadata['id']}/{version}" if version else metadata["id"]
        return PushResult(ref, version, output.strip())

    def status(self, ref: str) -> StatusResult:
        # UNVERIFIED: that the service accepts a version-suffixed ref, owner/slug/3.
        return parse_status(self._run(["kernels", "status", ref], timeout=60))

    def output(self, ref: str, dest: Path) -> OutputResult:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        # UNVERIFIED: outputs and the <slug>.log file can be downloaded, also for a failed kernel.
        self._run(["kernels", "output", ref, "-p", str(dest), "-o"], timeout=900)
        slug = ref.split("/")[1]
        log_file = dest / f"{slug}.log"
        if log_file.is_file():
            raw = log_file.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                raw = self._run(["kernels", "logs", ref], timeout=120)
            except KaggleError:
                raw = ""
        files = [p for p in dest.rglob("*") if p.is_file()]
        return OutputResult(dest, files, log_to_text(raw.strip()))
