"""The whole loop offline: lock -> dispatch (local backend) -> watcher -> scorecard -> verdict."""

import json
import subprocess
from pathlib import Path

import pytest
from labtestkit import (
    TOY,
    RecordingNotifier,
    approve_and_lock,
    copy_lab,
    make_watcher,
    replace_pipeline,
    runs,
    set_gate_approval,
    tool,
)

from labcore import registry
from labcore.io import utcnow
from labcore.kaggle.local_backend import LocalBackend
from labcore.lab import Lab
from labcore.notify.github import GitHubNotifier
from labcore.runs.dispatch import dispatch
from labcore.runs.state import StateStore
from labcore.stats.scorecard import scorecard_problems
from labcore.verdict.verdict import BEGIN, END, INTERPRETATION
from labcore.watcher import watcher as watching


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_valid_verdict_md(path: Path, verdict: str):
    text = path.read_text(encoding="utf-8")
    assert text.count(BEGIN) == text.count(END) == text.count(INTERPRETATION) == 1
    assert text.index(BEGIN) < text.index(END) < text.index(INTERPRETATION)
    assert f"## Verdict: {verdict}" in text


def run_loop(root: Path, slug: str, exp: str, backend: LocalBackend):
    lab = Lab.at(root)
    outcome = dispatch(lab, slug, exp, backend, require_lock=True)
    assert outcome.action == "submitted"
    notifier = RecordingNotifier()
    watcher = make_watcher(root, {"local": backend}, notifier, clock=utcnow)
    report = watcher.run_once()
    assert watcher.errors == [], report
    return outcome.record["run_id"], notifier


def test_toy_loop_ends_in_go(toy_lab, tmp_path):
    lock_tool = tool("lock_prereg")
    passed, message = tool("validate").run_l0(toy_lab, TOY)
    assert passed, message
    set_gate_approval(toy_lab, TOY)
    lock_tool.lock(toy_lab, TOY)

    run_id, notifier = run_loop(toy_lab, TOY, "exp_001", LocalBackend(tmp_path / "kernels"))

    [run] = runs(toy_lab)
    folder = toy_lab / run["result_path"]
    assert run["status"] == "COMPLETE" and run["failure_class"] is None
    assert scorecard_problems(read(folder / "scorecard.json")) == []
    assert read(folder / "verdict.json")["verdict"] == "GO"
    assert read(folder / "checks.json") == {"problems": []}
    [issue] = notifier.issues.values()
    assert issue["title"] == f"[tracking] {TOY}" and "Verdict (computed by script" in issue["body"]
    assert "# Verdict" in (toy_lab / "directions" / TOY / "VERDICT.md").read_text()
    # the watcher itself records nothing in the registry or VERDICT.md
    assert registry.load(toy_lab / "registry.yaml")["directions"][TOY]["status"] == "L1_RUNNING"
    budget = StateStore(toy_lab / "state").budget()
    assert budget["entries"] == [] or budget["entries"][0]["settled"]

    verdict_tool = tool("verdict")
    lab = Lab.at(toy_lab)
    with pytest.raises(verdict_tool.VerdictError, match="no usable run"):
        verdict_tool.record_verdict(lab, TOY)  # a local run is not evidence
    assert verdict_tool.record_verdict(lab, TOY, run_id, allow_local=True).verdict == "GO"
    assert_valid_verdict_md(toy_lab / "directions" / TOY / "VERDICT.md", "GO")
    item = registry.load(toy_lab / "registry.yaml")["directions"][TOY]
    assert (item["status"], item["verdict"]) == ("VERDICT", "GO")
    assert tool("validate").validate_lab(toy_lab) == []


def test_a_raising_kernel_ends_in_error_with_an_issue(toy_lab, tmp_path):
    replace_pipeline(toy_lab, "print('about to fail')\nraise ValueError('shapes differ')")
    approve_and_lock(toy_lab, TOY)
    backend = LocalBackend(tmp_path / "kernels")
    dispatch(Lab.at(toy_lab), TOY, "exp_001", backend, require_lock=True)

    calls = []

    def fake_gh(command, **options):
        calls.append((command[1:], options.get("input")))
        out = "https://github.com/o/r/issues/5\n" if command[2] == "create" else "[]"
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")

    # the real diagnose function: no DEEPSEEK_API_KEY is set, so nothing is sent
    watcher = watching.Watcher(
        Lab.at(toy_lab), {"local": backend}, GitHubNotifier(runner=fake_gh, env={}),
        send_telegram=lambda text: False,
    )
    watcher.run_once()

    [run] = runs(toy_lab)
    folder = toy_lab / run["result_path"]
    assert (run["status"], run["failure_class"], run["issue_number"]) == ("ERROR", "code", 5)
    log = (folder / "error_log.txt").read_text()
    assert "about to fail" in log and "ValueError: shapes differ" in log
    assert (folder / "diagnosis.md").read_text().startswith("diagnosis unavailable")
    assert not (folder / "scorecard.json").exists()
    [(arguments, body)] = [call for call in calls if call[0][:2] == ["issue", "create"]]
    assert "run-failed" in arguments
    assert "diagnosis unavailable" in body and "shapes differ" in body


@pytest.mark.parametrize(
    "exception, failure_class, total_runs",
    [("ConnectionError", "infra", 2), ("ValueError", "code", 1)],
)
def test_retry_policy_end_to_end(toy_lab, tmp_path, exception, failure_class, total_runs):
    replace_pipeline(toy_lab, f"raise {exception}('simulated')")
    approve_and_lock(toy_lab, TOY)
    backend = LocalBackend(tmp_path / "kernels")
    dispatch(Lab.at(toy_lab), TOY, "exp_001", backend, require_lock=True)
    notifier = RecordingNotifier()
    watcher = make_watcher(toy_lab, {"local": backend}, notifier, clock=utcnow)
    for _ in range(4):
        watcher.run_once()
    records = runs(toy_lab)
    assert len(records) == total_runs
    assert {r["status"] for r in records} == {"ERROR"}
    assert {r["failure_class"] for r in records} == {failure_class}
    assert [r["retries"] for r in records] == list(range(total_runs))
    assert len(notifier.issues) == 1 and watcher.errors == []


def test_failed_direction_checks_make_the_run_invalid(toy_lab, tmp_path):
    write = toy_lab / "directions" / TOY / "checks.py"
    write.write_text("def check(result_dir):\n    return ['deliberately failing check']\n")
    approve_and_lock(toy_lab, TOY)
    _, notifier = run_loop(toy_lab, TOY, "exp_001", LocalBackend(tmp_path / "kernels"))
    [run] = runs(toy_lab)
    assert (run["status"], run["failure_class"]) == ("COMPLETE_INVALID", "code")
    assert not (toy_lab / run["result_path"] / "scorecard.json").exists()
    assert "deliberately failing check" in list(notifier.issues.values())[0]["body"]


def test_dry_run_commands(toy_lab, capsys):
    from labcore.runs import dispatch as dispatching

    root = str(toy_lab)
    assert dispatching.main([TOY, "exp_001", "--dry-run", "--root", root]) == 0
    assert watching.main(["--dry-run", "--root", root]) == 0
    assert tool("verdict").main([TOY, "--dry-run", "--root", root]) == 0
    output = capsys.readouterr().out
    assert "dry run: GO" in output
    preview = (toy_lab / ".dryrun" / f"VERDICT.{TOY}.md").read_text(encoding="utf-8")
    assert "This is not a result" in preview
    assert not (toy_lab / "state" / "runs.json").exists()
    assert list((toy_lab / "results").iterdir()) == []


def test_isles_loop_with_fake_data(tmp_path):
    """Direction #1 from L0 to VERDICT.md, on a copy of the lab whose gates are approved."""
    for module in ("torch", "nibabel", "pyarrow", "scipy"):
        pytest.importorskip(module)
    slug = "isles24-ctp"
    root = copy_lab(tmp_path / "lab", (slug,))
    registry.set_status(root, slug, "SCAFFOLDED", l0=None)
    passed, message = tool("validate").run_l0(root, slug)
    assert passed, message
    set_gate_approval(root, slug)  # only in this throwaway copy
    tool("lock_prereg").lock(root, slug)

    from labcore.config import load_experiment
    from labcore.modules import load_module

    directory = root / "directions" / slug
    fake = load_module(directory / "tests" / "fake_data.py")
    base = load_experiment(directory / "configs" / "exp_001_pilot.yaml")
    config = fake.make_dryrun_config(tmp_path / "fake", base)
    (directory / "configs" / "exp_900_fake.yaml").write_text(json.dumps(config), encoding="utf-8")

    run_id, notifier = run_loop(root, slug, "exp_900_fake", LocalBackend(tmp_path / "kernels"))

    [run] = runs(root)
    folder = root / run["result_path"]
    assert run["status"] == "COMPLETE", (folder / "checks.json").read_text()
    card = read(folder / "scorecard.json")
    assert scorecard_problems(card) == []
    assert (card["n_subjects_requested"], card["n_subjects_usable"]) == (6, 6)
    assert {"mean(raw-maps)", "ci_hi(raw-maps)", "ci_lo(raw-N1)", "mde149(raw-maps)"} <= set(
        card["metrics"]
    )
    assert not list(folder.glob("*.png"))  # patient images are never committed
    expected = read(folder / "verdict.json")["verdict"]

    result = tool("verdict").record_verdict(Lab.at(root), slug, run_id, allow_local=True)
    assert result.verdict == expected
    assert_valid_verdict_md(directory / "VERDICT.md", expected)
