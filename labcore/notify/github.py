"""GitHub issues and comments through the `gh` CLI.

In Actions, `gh` authenticates with GH_TOKEN / GITHUB_TOKEN and takes the repository from the git
remote of the working folder. Notification failures are printed and swallowed: they must not lose
state.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from typing import Protocol

MAX_BODY = 60000  # GitHub rejects bodies over 65536 characters


def fenced(text: str) -> str:
    """Wrap untrusted text in a code fence it cannot break out of (this also mutes @mentions)."""
    longest = max((len(run) for run in re.findall(r"~{3,}", text)), default=2)
    fence = "~" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


class Notifier(Protocol):
    def find_issue(self, title: str) -> int | None: ...

    def create_issue(self, title: str, body: str, labels: tuple[str, ...] = ()) -> int | None: ...

    def comment(self, number: int, body: str) -> bool: ...


class GitHubNotifier:
    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        env: Mapping[str, str] | None = None,
    ):
        self.runner = runner
        self.env = dict(os.environ if env is None else env)

    def _gh(self, *args: str, body: str | None = None) -> str | None:
        env = dict(self.env)
        if not env.get("GH_TOKEN") and env.get("GITHUB_TOKEN"):
            env["GH_TOKEN"] = env["GITHUB_TOKEN"]
        try:
            done = self.runner(
                [shutil.which("gh") or "gh", *args], input=body, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60, env=env, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"notify: gh {' '.join(args[:2])} failed: {type(exc).__name__}")
            return None
        if done.returncode != 0:
            print(f"notify: gh {' '.join(args[:2])} failed: {(done.stderr or '').strip()[:500]}")
            return None
        return done.stdout or ""

    def find_issue(self, title: str) -> int | None:
        """The number of an issue with exactly this title, so repeats do not open duplicates."""
        # UNVERIFIED: GitHub search finds a just-created issue by title (indexing may lag).
        output = self._gh(
            "issue", "list", "--state", "all", "--limit", "50", "--search", f'in:title "{title}"',
            "--json", "number,title",
        )
        if not output:
            return None
        try:
            matches = [item["number"] for item in json.loads(output) if item.get("title") == title]
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
        return min(matches) if matches else None

    def create_issue(self, title: str, body: str, labels: tuple[str, ...] = ()) -> int | None:
        args = ["issue", "create", "--title", title, "--body-file", "-"]
        labelled = [*args, *(part for label in labels for part in ("--label", label))]
        output = self._gh(*labelled, body=body[:MAX_BODY])
        if output is None and labels:  # most likely the label does not exist yet
            output = self._gh(*args, body=body[:MAX_BODY])
        match = re.search(r"/issues/(\d+)", output or "")
        return int(match.group(1)) if match else None

    def comment(self, number: int, body: str) -> bool:
        return self._gh("issue", "comment", str(number), "--body-file", "-", body=body[:MAX_BODY]) is not None


class ConsoleNotifier:
    """Used for dry runs and when notifications are switched off: prints instead of posting."""

    def find_issue(self, title: str) -> int | None:
        return None

    def create_issue(self, title: str, body: str, labels: tuple[str, ...] = ()) -> int | None:
        print(f"[notify:off] would open issue {title!r} labels={list(labels)}\n{body}\n")
        return None

    def comment(self, number: int, body: str) -> bool:
        print(f"[notify:off] would comment on issue #{number}\n{body}\n")
        return True
