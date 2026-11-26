"""Failure classification, and the watcher against a fake Kaggle (status set by the test)."""

import pytest
from labtestkit import (
    TOY,
    Clock,
    FakeKaggle,
    RecordingNotifier,
    approve_and_lock,
    make_watcher,
    runs,
)

from labcore.kaggle.client import KaggleError, KernelStatus, OutputResult
from labcore.lab import Lab
from labcore.runs.dispatch import dispatch
from labcore.runs.state import StateStore
from labcore.watcher.checks import copy_results, direction_checks, generic_checks
from labcore.watcher.classify import classify

TRACEBACK = "Traceback (most recent call last):\n  File 'x.py', line 1, in <module>\n"


def failed(error_type, message=""):
    return {"error_type": error_type, "error_message": message}


@pytest.mark.parametrize(
    "arguments",
    [
        dict(status_json=failed("ConnectionError", "boom")),
        dict(status_json=failed("RuntimeError", "pip install failed: Max retries exceeded with url")),
        dict(status_json=failed("HfHubHTTPError", "503 Server Error: Service Unavailable")),
        dict(status_json=failed("RuntimeError", "too few usable cases: mostly download failed (30 of 31)")),
        dict(log_text=TRACEBACK + "requests.exceptions.ReadTimeout: Read timed out."),
        dict(log_text="", failure_message="Session terminated"),
        dict(log_text=""),  # died before our code wrote anything
        dict(run_status="TIMEOUT", last_seen="QUEUED"),
        dict(run_status="COMPLETE_INVALID", check_failures=["3/40 usable; mostly download failed (37)"]),
    ],
)
def test_infrastructure_failures(arguments):
    arguments.setdefault("run_status", "ERROR")
    assert classify(**arguments).failure_class == "infra"


@pytest.mark.parametrize(
    "arguments",
    [
        dict(status_json=failed("ValueError", "shapes differ")),
        dict(status_json=failed("HfHubHTTPError", "401 Client Error: Unauthorized for url")),
        dict(status_json=failed("RuntimeError", "download failed: GatedRepoError 403 Client Error")),
        dict(status_json=failed("RuntimeError", "CUDA out of memory")),
        # an earlier network warning does not outweigh the final exception
        dict(status_json=failed("KeyError", "'dice'"), log_text="Connection reset by peer, retrying"),
        dict(log_text="Connection reset by peer\n" + TRACEBACK + "KeyError: 'dice'"),
        dict(log_text="CUDA out of memory"),
        dict(run_status="TIMEOUT", last_seen="RUNNING"),
        dict(run_status="CANCELLED", failure_message="Session terminated"),
        dict(run_status="COMPLETE_INVALID", check_failures=["Dice is zero for every patient in arm raw"]),
        dict(run_status="ERROR", status_json={"error_type": None}, log_text="finished with exit code 1"),
    ],
)
def test_code_failures(arguments):
    arguments.setdefault("run_status", "ERROR")
    assert classify(**arguments).failure_class == "code"


# ---------------------------------------------------------------- checks


def test_only_declared_small_outputs_are_copied(tmp_path):
    out, results = tmp_path / "out", tmp_path / "results"
    out.mkdir()
    for name in ("status.json", "env.json", "summary.json", "sanity.png", "extra.txt", "weights.pt"):
        (out / name).write_text("{}")
    (out / "huge.csv").write_bytes(b"x" * (6 * 1024 * 1024))
    problems = copy_results(out, results, ["summary.json", "weights.pt", "huge.csv", "../outside.txt"])
    assert sorted(p.name for p in results.iterdir()) == ["env.json", "status.json", "summary.json"]
    assert len(problems) == 2


def test_generic_checks(tmp_path):
    (tmp_path / "status.json").write_text('{"ok": false, "stage": "run"}')
    (tmp_path / "empty.csv").write_text("")
    problems = generic_checks(tmp_path, ["empty.csv", "absent.csv"])
    assert len(problems) == 3


def test_a_crashing_check_is_a_failed_check(tmp_path):
    (tmp_path / "checks.py").write_text("def check(result_dir):\n    raise KeyError('x')\n")
    assert "crashed" in direction_checks(tmp_path / "checks.py", tmp_path)[0]
    assert direction_checks(tmp_path / "absent.py", tmp_path) == []


# ---------------------------------------------------------------- watcher


@pytest.fixture
def submitted(toy_lab):
    approve_and_lock(toy_lab, TOY)
    clock, kaggle = Clock(), FakeKaggle()
    dispatch(Lab.at(toy_lab), TOY, "exp_001", kaggle, username="tester", now=clock())
    notifier = RecordingNotifier()
    watcher = make_watcher(toy_lab, {"kaggle": kaggle}, notifier, clock, diagnose=lambda t, s: "a summary")
    return toy_lab, kaggle, clock, notifier, watcher


def test_running_runs_are_only_updated(submitted):
    root, kaggle, clock, notifier, watcher = submitted
    kaggle.current = KernelStatus.RUNNING
    clock.advance(minutes=30)
    watcher.run_once()
    [run] = runs(root)
    assert (run["status"], run["last_checked_at"]) == ("RUNNING", "2026-11-26T12:30:00Z")
    assert run["finished_at"] is None and notifier.issues == {} and watcher.errors == []


def test_timeout_after_twice_the_estimate(submitted):
    root, kaggle, clock, notifier, watcher = submitted
    kaggle.current = KernelStatus.RUNNING
    clock.advance(minutes=60)
    watcher.run_once()
    assert runs(root)[0]["status"] == "RUNNING"
    clock.advance(minutes=1)
    watcher.run_once()
    [run] = runs(root)
    assert (run["status"], run["failure_class"]) == ("TIMEOUT", "code")
    assert len(kaggle.pushes) == 1
    assert run["actual_gpu_hours"] == pytest.approx(61 / 60, abs=1e-3)
    assert list(notifier.issues.values())[0]["labels"] == ("run-failed",)


def test_a_kernel_stuck_in_the_queue_is_retried_exactly_once(submitted):
    root, kaggle, clock, notifier, watcher = submitted
    clock.advance(hours=2)
    watcher.run_once()
    first, second = runs(root)
    assert (first["status"], first["failure_class"]) == ("TIMEOUT", "infra")
    assert first["actual_gpu_hours"] == 0  # it never started, so it used no GPU time
    assert (second["status"], second["retries"], second["retry_of"]) == ("QUEUED", 1, first["run_id"])
    assert second["issue_number"] == first["issue_number"] == 1
    assert second["prereg_hash"] == first["prereg_hash"]

    clock.advance(hours=2)
    watcher.run_once()
    assert len(runs(root)) == 2 and len(kaggle.pushes) == 2
    assert runs(root)[1]["status"] == "TIMEOUT"
    assert len(notifier.issues) == 1 and "No further retry" in notifier.comments[-1][1]


@pytest.mark.parametrize(
    "log, failure_class, pushes",
    [(TRACEBACK + "ConnectionError: reset", "infra", 2), (TRACEBACK + "TypeError: bad", "code", 1)],
)
def test_only_infrastructure_errors_are_retried(submitted, log, failure_class, pushes):
    root, kaggle, clock, _, watcher = submitted
    kaggle.current, kaggle.log = KernelStatus.ERROR, log
    clock.advance(minutes=10)
    watcher.run_once()
    assert runs(root)[0]["failure_class"] == failure_class
    assert len(kaggle.pushes) == pushes


def test_no_retry_when_the_code_changed_since_submission(submitted):
    root, kaggle, clock, notifier, watcher = submitted
    with open(root / "directions" / TOY / "src" / "main.py", "a", encoding="utf-8") as handle:
        handle.write("# changed\n")
    kaggle.current = KernelStatus.ERROR
    watcher.run_once()
    assert len(kaggle.pushes) == 1
    assert "not submitted" in list(notifier.issues.values())[0]["body"]


def test_no_retry_beyond_the_budget(submitted):
    root, kaggle, clock, notifier, watcher = submitted
    store = StateStore(root / "state")
    budget = store.budget()
    budget["weekly_cap_hours"] = 0.4
    store.save_budget(budget)
    kaggle.current = KernelStatus.ERROR
    clock.advance(hours=1)
    watcher.run_once()
    assert len(kaggle.pushes) == 1 and "weekly GPU cap" in list(notifier.issues.values())[0]["body"]


def test_failure_report_files_and_issue(submitted):
    root, kaggle, clock, notifier, watcher = submitted
    secret = "hf_" + "A1" * 15
    kaggle.current = KernelStatus.ERROR
    kaggle.log = "\n".join(f"line {i}" for i in range(500)) + f"\nHF_TOKEN={secret}\n" + TRACEBACK + "X: y\n"
    watcher.run_once()
    [run] = runs(root)
    folder = root / run["result_path"]
    log = (folder / "error_log.txt").read_text()
    assert len(log.splitlines()) == 200 and "line 499" in log and "line 100\n" not in log
    assert secret not in log
    assert (folder / "diagnosis.md").read_text() == "a summary\n"
    [issue] = notifier.issues.values()
    assert issue["title"] == f"[run-failed] {TOY}/exp_001 {run['run_id']}"
    for text in ("ERROR", "**code**", "a summary", "line 499"):
        assert text in issue["body"]
    assert secret not in issue["body"] and run["issue_number"] == 1


def test_cancelled_runs_are_reported_but_not_retried(submitted):
    root, kaggle, _, notifier, watcher = submitted
    kaggle.current, kaggle.output_error = KernelStatus.CANCELLED, True
    watcher.run_once()
    [run] = runs(root)
    assert run["status"] == "CANCELLED" and len(kaggle.pushes) == 1 and len(notifier.issues) == 1


def test_logs_of_an_earlier_kernel_version_are_ignored(submitted):
    root, kaggle, _, _, watcher = submitted
    kaggle.current = KernelStatus.ERROR

    def stale(ref, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "status.json").write_text('{"run_id": "an-older-run", "error_type": "TypeError"}')
        return OutputResult(dest, [], TRACEBACK + "TypeError: from the older version")

    kaggle.output = stale
    watcher.run_once()
    first = runs(root)[0]
    assert first["failure_class"] == "infra"  # nothing from this run: not judged by old logs
    log = (root / first["result_path"] / "error_log.txt").read_text()
    assert "older version" not in log and "another run" in log


def test_complete_without_outputs_is_invalid(submitted):
    root, kaggle, _, notifier, watcher = submitted
    kaggle.current = KernelStatus.COMPLETE
    watcher.run_once()
    [run] = runs(root)
    assert (run["status"], run["failure_class"]) == ("COMPLETE_INVALID", "code")
    assert "status.json is missing" in list(notifier.issues.values())[0]["body"]
    assert not (root / run["result_path"] / "scorecard.json").exists()


def test_a_failed_download_is_tried_again(submitted):
    root, kaggle, clock, notifier, watcher = submitted
    kaggle.current, kaggle.output_error = KernelStatus.COMPLETE, True
    clock.advance(minutes=20)
    watcher.run_once()
    assert runs(root)[0]["status"] == "QUEUED" and notifier.issues == {}
    clock.advance(hours=25)
    watcher.run_once()
    assert runs(root)[0]["status"] == "ERROR"


def test_status_errors_do_not_end_a_run_early(submitted):
    root, kaggle, clock, _, watcher = submitted

    def broken(ref):
        raise KaggleError("503")

    kaggle.status = broken
    watcher.run_once()
    assert runs(root)[0]["status"] == "QUEUED" and watcher.errors == []


def test_a_second_pass_does_nothing(submitted):
    root, kaggle, _, notifier, watcher = submitted
    kaggle.current = KernelStatus.ERROR
    kaggle.log = TRACEBACK + "TypeError: bad"
    watcher.run_once()
    def contents():
        watched = [*(root / "state").rglob("*"), *(root / "results").rglob("*")]
        return {p: p.read_bytes() for p in watched if p.is_file()}

    files = contents()
    announced = (len(notifier.issues), len(notifier.comments))
    assert watcher.run_once() == []
    assert contents() == files
    assert (len(notifier.issues), len(notifier.comments)) == announced


def test_the_watcher_writes_only_state_and_results(submitted):
    root, kaggle, _, _, watcher = submitted

    def snapshot():
        return {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }

    before = snapshot()
    kaggle.current = KernelStatus.ERROR
    watcher.run_once()
    after = snapshot()
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    assert changed and all(name.startswith(("state/", "results/")) for name in changed)


def test_one_broken_run_does_not_block_the_others(submitted):
    root, kaggle, clock, _, watcher = submitted
    store = StateStore(root / "state")
    good = store.runs()[0]
    store.save_runs([{**good, "run_id": "broken", "backend": "local"}, good])
    kaggle.current = KernelStatus.RUNNING
    watcher.run_once()
    assert len(watcher.errors) == 1 and runs(root)[1]["status"] == "RUNNING"
