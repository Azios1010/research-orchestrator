"""Poll active runs, collect results, check them and report. Safe to run repeatedly.

    python -m labcore.watcher.watcher             # real runs (state/, results/, GitHub issues)
    python -m labcore.watcher.watcher --dry-run   # local runs under .dryrun/, prints notifications

Only the state and results folders are ever written. A run's record is made terminal and saved
before anything is announced, so a second pass never processes it again.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from labcore import registry
from labcore.config import ConfigError, load_direction
from labcore.diagnose import deepseek
from labcore.gitutil import extract_commit
from labcore.io import hours_between, iso, read_json, tail_lines, utcnow, write_json, write_text_atomic
from labcore.kaggle.bundle import BundleError
from labcore.kaggle.client import Backend, KaggleClient, KaggleError, KernelStatus
from labcore.kaggle.local_backend import LocalBackend
from labcore.lab import DRYRUN_DIRNAME, Lab, find_root
from labcore.notify import telegram
from labcore.notify.github import ConsoleNotifier, GitHubNotifier, Notifier, fenced
from labcore.privacy import names_in
from labcore.redact import redact_secrets
from labcore.runs import state as st
from labcore.runs.dispatch import DispatchError, dispatch
from labcore.stats.scorecard import (
    ScorecardError,
    build_scorecard,
    flatten,
    runtime_from_status,
    scorecard_problems,
)
from labcore.verdict.gates import GateError
from labcore.verdict.prereg import LOCKED_TEXT, matches_stored
from labcore.verdict.verdict import VerdictError, compute_verdict
from labcore.watcher.checks import copy_results, direction_checks, generic_checks
from labcore.watcher.classify import classify

LOG_TAIL_LINES = 200
ISSUE_LOG_LINES = 80
MAX_RETRIES = 1
STALE_OUTPUT_GRACE_HOURS = 20 / 60
DOWNLOAD_GIVE_UP_HOURS = 24.0
FAILED_LABEL = "run-failed"
PENDING = (KernelStatus.QUEUED, KernelStatus.RUNNING, KernelStatus.UNKNOWN)


class Watcher:
    def __init__(
        self,
        lab: Lab,
        backends: Mapping[str, Backend],
        notifier: Notifier,
        *,
        diagnose: Callable[..., str] = deepseek.diagnose,
        send_telegram: Callable[[str], bool] = telegram.send,
        now: Callable[[], datetime] = utcnow,
    ):
        self.lab = lab
        self.backends = backends
        self.notifier = notifier
        self.diagnose = diagnose
        self.send_telegram = send_telegram
        self.now = now
        self.store = st.StateStore(lab.state_dir)
        self.errors: list[str] = []

    def run_once(self) -> list[str]:
        report = []
        for run in self.store.active_runs():
            try:
                report.append(self._check(run))
            except Exception as exc:  # one broken run must not block the others
                message = f"{run['run_id']}: watcher error: {type(exc).__name__}: {exc}"
                self.errors.append(redact_secrets(message))
                report.append(self.errors[-1])
        return report

    # ------------------------------------------------------------------ polling

    def _check(self, run: dict) -> str:
        backend = self.backends[run["backend"]]
        version = run.get("kernel_version")
        ref = f"{run['kernel_id']}/{version}" if version else run["kernel_id"]
        now = self.now()
        elapsed = hours_between(run["submitted_at"], now)
        try:
            result = backend.status(ref)
        except KaggleError as exc:
            result, problem = None, str(exc)

        if result is None or result.status in PENDING:
            seen = result.status.value if result else run["status"]
            if elapsed > st.timeout_hours(run["est_gpu_hours"]):
                queued = result is not None and result.status == KernelStatus.QUEUED
                return self._fail(run, "TIMEOUT", backend, ref, last_seen=seen, never_started=queued)
            run.update(status=seen, last_checked_at=iso(now))
            self.store.put_run(run)
            note = f" (status check failed: {redact_secrets(problem)[:200]})" if result is None else ""
            return f"{run['run_id']}: {seen} after {elapsed:.2f} h{note}"
        if result.status == KernelStatus.COMPLETE:
            return self._complete(run, backend, ref)
        status = "CANCELLED" if result.status == KernelStatus.CANCELLED else "ERROR"
        return self._fail(run, status, backend, ref, failure_message=result.failure_message)

    def _download(self, backend: Backend, ref: str, destination: Path) -> tuple[Path, str]:
        output = backend.output(ref, destination)
        for candidate in (destination / "out", destination):
            if (candidate / "status.json").is_file():
                return candidate, output.log_text
        found = sorted(destination.rglob("status.json"))
        return (found[0].parent if found else destination), output.log_text

    def _source_for(self, run: dict, scratch: Path) -> Path:
        """The checkout the run was submitted from, when git can provide it; else this one."""
        sha, slug = run["commit_sha"], run["slug"]
        if sha != "unknown" and not sha.endswith("-dirty"):
            paths = ["labcore", f"directions/{slug}", "registry.yaml"]
            if extract_commit(self.lab.root, sha, paths, scratch):
                return scratch
        return self.lab.root

    # ------------------------------------------------------------------ success

    def _complete(self, run: dict, backend: Backend, ref: str) -> str:
        elapsed = hours_between(run["submitted_at"], self.now())
        with tempfile.TemporaryDirectory(prefix="lab_watch_") as temporary:
            temporary = Path(temporary)
            try:
                out, log_text = self._download(backend, ref, temporary / "download")
            except KaggleError as exc:
                if elapsed > DOWNLOAD_GIVE_UP_HOURS:
                    return self._fail(run, "ERROR", backend, ref, failure_message=f"download failed: {exc}")
                run["last_checked_at"] = iso(self.now())
                self.store.put_run(run)
                return f"{run['run_id']}: finished, but the download failed; will try again"

            env = _safe_json(out / "env.json") or _safe_json(out / "status.json") or {}
            if env.get("run_id") not in (None, run["run_id"]):
                if elapsed < STALE_OUTPUT_GRACE_HOURS:
                    return f"{run['run_id']}: outputs of an older version are still being served; waiting"
                return self._fail(
                    run, "COMPLETE_INVALID", backend, ref, log_text=log_text,
                    check_failures=[f"downloaded outputs belong to run {env['run_id']}"],
                )

            direction_dir = self._source_for(run, temporary / "source") / "directions" / run["slug"]
            direction = load_direction(direction_dir)
            result_dir = self.lab.results_dir / run["slug"] / run["exp"] / run["run_id"]
            columns = direction.get("results_columns")
            problems = copy_results(out, result_dir, direction["outputs"], columns)
            problems += generic_checks(result_dir, direction["outputs"], columns)
            scorecard = None
            if not problems:
                try:
                    scorecard = build_scorecard(result_dir, direction["scorecard_map"], run, pipeline_ok=True)
                    problems += scorecard_problems(scorecard)
                    if found := names_in(scorecard):  # scorecard.json is committed too
                        problems.append(f"scorecard has phenotype-like field(s): {', '.join(found)}")
                except (ScorecardError, OSError, ValueError, KeyError) as exc:
                    problems.append(f"scorecard could not be built: {exc}")
                problems += direction_checks(direction_dir / "checks.py", result_dir)
            write_json(result_dir / "checks.json", {"problems": problems})
            if problems:
                return self._fail(
                    run, "COMPLETE_INVALID", backend, ref, log_text=log_text, check_failures=problems,
                    status_json=_safe_json(result_dir / "status.json"),
                )

            write_json(result_dir / "scorecard.json", scorecard)
            verdict, verdict_note = self._verdict(run, direction_dir, direction, scorecard)
            if verdict:
                write_json(
                    result_dir / "verdict.json",
                    {**verdict, "prereg_hash": run["prereg_hash"], "provisional": True},
                )

        runtime_min, _ = runtime_from_status(_safe_json(result_dir / "status.json"), None)
        self._finish(run, "COMPLETE", runtime_min / 60)
        return self._announce_success(run, scorecard, verdict_note)

    def _verdict(self, run: dict, direction_dir: Path, direction: dict, scorecard: dict):
        """Only for L1 runs, and only with exactly what was locked at submission."""
        if not run.get("prereg_hash"):
            return None, "No verdict: this run was submitted without a pre-registration lock."
        try:
            unchanged = matches_stored(direction_dir, run["prereg_hash"])
        except (OSError, ValueError) as exc:
            return None, f"No verdict: the lock cannot be checked ({exc})."
        if direction["gates"]["approval"] != "APPROVED" or not unchanged:
            return None, f"No verdict: what was locked ({LOCKED_TEXT}) differs from what is available now."
        try:
            result = compute_verdict(direction["gates"], scorecard)
        except (GateError, VerdictError) as exc:
            return None, f"No verdict: {exc}"
        return result.as_dict(), (
            f"**Verdict (computed by script, provisional): {result.verdict}**. Record it with "
            f"`python tools/verdict.py {run['slug']}` in a pull request."
        )

    def _announce_success(self, run: dict, scorecard: dict, verdict_note: str) -> str:
        rows = "\n".join(f"| `{name}` | {value} |" for name, value in flatten(scorecard).items())
        body = (
            f"### Run `{run['run_id']}` completed\n\n"
            f"Experiment `{run['exp']}`, commit `{run['commit_sha']}`, results in "
            f"`{run['result_path']}`.\n\n| Field | Value |\n|---|---|\n{rows}\n\n{verdict_note}\n"
        )
        title = f"[tracking] {run['slug']}"
        number = self.notifier.find_issue(title)
        if number:
            self.notifier.comment(number, body)
        else:
            number = self.notifier.create_issue(title, body)
        if run.get("issue_number"):  # a retry that succeeded: close the loop on the failure issue
            self.notifier.comment(run["issue_number"], f"Retry `{run['run_id']}` completed successfully.")
        elif number:
            run["issue_number"] = number
            self.store.put_run(run)
        self.send_telegram(f"{run['slug']}/{run['exp']}: COMPLETE ({run['run_id']}). {verdict_note}")
        return f"{run['run_id']}: COMPLETE"

    # ------------------------------------------------------------------ failure

    def _finish(self, run: dict, status: str, hours: float, failure_class: str | None = None) -> None:
        now = self.now()
        run.update(
            status=status, finished_at=iso(now), last_checked_at=iso(now),
            actual_gpu_hours=round(hours, 4), failure_class=failure_class,
        )
        self.store.put_run(run)
        budget = self.store.budget()
        st.settle(budget, run["run_id"], run["slug"], hours, run["finished_at"], now)
        self.store.save_budget(budget)

    def _fail(
        self,
        run: dict,
        status: str,
        backend: Backend,
        ref: str,
        *,
        log_text: str | None = None,
        status_json: dict | None = None,
        failure_message: str | None = None,
        check_failures: list[str] | None = None,
        last_seen: str | None = None,
        never_started: bool = False,
    ) -> str:
        result_dir = self.lab.results_dir / run["slug"] / run["exp"] / run["run_id"]
        result_dir.mkdir(parents=True, exist_ok=True)
        if never_started:  # nothing ran; whatever Kaggle serves belongs to an earlier version
            log_text = ""
        if log_text is None:  # not downloaded yet
            log_text = ""
            with tempfile.TemporaryDirectory(prefix="lab_fail_") as temporary:
                note = None
                try:
                    out, log_text = self._download(backend, ref, Path(temporary))
                    found = _safe_json(out / "status.json")
                    if found and found.get("run_id") == run["run_id"]:
                        status_json = found
                        copy_results(out, result_dir, [])
                    elif found:  # an earlier version's outputs, so its log as well
                        log_text, note = "", "the downloaded outputs and log belong to another run"
                except KaggleError as exc:
                    note = f"logs could not be downloaded: {exc}"
                if note:
                    failure_message = f"{failure_message or ''}\n{note}".strip()

        tail = redact_secrets(tail_lines(log_text, LOG_TAIL_LINES))
        if not tail.strip():
            pieces = [failure_message, (status_json or {}).get("traceback_tail"), *(check_failures or [])]
            tail = redact_secrets("\n".join(p for p in pieces if p)) or "(no log available)"
        write_text_atomic(result_dir / "error_log.txt", tail + "\n")

        verdict_class = classify(
            run_status=status, status_json=status_json, log_text=tail, failure_message=failure_message,
            check_failures=check_failures, last_seen=last_seen,
        )
        elapsed = hours_between(run["submitted_at"], self.now())
        runtime_min, _ = runtime_from_status(status_json, None)
        if status_json and status_json.get("finished_utc"):
            hours = runtime_min / 60
        elif never_started:
            hours = 0.0
        elif status == "TIMEOUT":
            hours = min(elapsed, st.HARD_TIMEOUT_HOURS)
        else:
            hours = min(elapsed, run["est_gpu_hours"])
        self._finish(run, status, hours, verdict_class.failure_class)

        retry, retry_note = None, "No automatic retry."
        if verdict_class.failure_class == "infra" and status != "CANCELLED":
            if run["retries"] < MAX_RETRIES:
                retry, retry_note = self._retry(run, backend)
            else:
                retry_note = "No further retry: this run was already the automatic retry."

        diagnosis = self.diagnose(tail, status_json)
        write_text_atomic(result_dir / "diagnosis.md", diagnosis.strip() + "\n")
        self._announce_failure(run, retry, verdict_class.reason, retry_note, tail, diagnosis, check_failures)
        return f"{run['run_id']}: {status} ({verdict_class.failure_class}). {retry_note}"

    def _retry(self, run: dict, backend: Backend) -> tuple[dict | None, str]:
        with tempfile.TemporaryDirectory(prefix="lab_retry_") as temporary:
            try:
                outcome = dispatch(
                    self.lab, run["slug"], run["exp"], backend,
                    username=run["kernel_id"].split("/")[0],
                    require_lock=run.get("prereg_hash") is not None,
                    source_root=self._source_for(run, Path(temporary)),
                    commit_sha=run["commit_sha"],
                    retries=run["retries"] + 1,
                    retry_of=run["run_id"],
                    issue_number=run.get("issue_number"),
                    expected_bundle_hash=run.get("bundle_hash"),
                    now=self.now(),
                )
            except (DispatchError, ConfigError, KaggleError, BundleError, registry.RegistryError) as exc:
                return None, f"Automatic retry not submitted: {redact_secrets(str(exc))[:500]}"
        if outcome.action != "submitted":
            return None, f"Automatic retry not submitted: {outcome.message}"
        return outcome.record, f"Retried automatically as `{outcome.record['run_id']}`."

    def _announce_failure(self, run, retry, reason, retry_note, tail, diagnosis, check_failures) -> None:
        checks = ""
        if check_failures:
            checks = "\n#### Failed checks\n" + fenced("\n".join(check_failures)) + "\n"
        body = (
            f"### Run `{run['run_id']}`: {run['status']}\n\n"
            f"- Direction / experiment: `{run['slug']}` / `{run['exp']}`\n"
            f"- Commit: `{run['commit_sha']}`\n"
            f"- Failure class: **{run['failure_class']}** ({reason})\n"
            f"- {retry_note}\n"
            f"- Files: `{run['result_path']}`\n{checks}\n"
            f"#### Diagnosis (automatic summary; may be wrong)\n{fenced(diagnosis)}\n\n"
            f"#### Log tail\n{fenced(tail_lines(tail, ISSUE_LOG_LINES))}\n"
        )
        title = f"[run-failed] {run['slug']}/{run['exp']} {run['run_id']}"
        number = run.get("issue_number") if run.get("retry_of") else None
        number = number or self.notifier.find_issue(title)
        if number:
            self.notifier.comment(number, body)
        else:
            number = self.notifier.create_issue(title, body, (FAILED_LABEL,))
        if number:
            run["issue_number"] = number
            self.store.put_run(run)
            if retry:
                retry["issue_number"] = number
                self.store.put_run(retry)
        self.send_telegram(
            f"{run['slug']}/{run['exp']}: {run['status']} ({run['failure_class']}), {run['run_id']}. "
            f"{retry_note}"
        )


def _safe_json(path: Path) -> dict | None:
    try:
        value = read_json(path)
    except (ValueError, OSError):
        return None
    return value if isinstance(value, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="watcher", description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="watch local runs under .dryrun/")
    parser.add_argument("--root", help="lab root (default: found from the current folder)")
    parser.add_argument("--state-dir")
    parser.add_argument("--results-dir")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    root = Path(args.root).resolve() if args.root else find_root()
    base = Lab.dry_run(root) if args.dry_run else Lab.at(root)
    lab = Lab.at(root, args.state_dir or base.state_dir, args.results_dir or base.results_dir)
    local = LocalBackend(root / DRYRUN_DIRNAME / "kernels")
    if args.dry_run:
        # A dry run talks to nobody: no GitHub, no DeepSeek, no Telegram.
        watcher = Watcher(
            lab, {"local": local}, ConsoleNotifier(),
            diagnose=lambda tail, status: f"{deepseek.UNAVAILABLE} (dry run)",
            send_telegram=lambda text: False,
        )
    else:
        watcher = Watcher(lab, {"kaggle": KaggleClient(), "local": local}, GitHubNotifier())
    report = watcher.run_once()
    for line in report or ["no active runs"]:
        print(line)
    return 1 if watcher.errors else 0


if __name__ == "__main__":
    sys.exit(main())
