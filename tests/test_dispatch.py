"""Budget, state files and dispatch (with a fake Kaggle backend: nothing is submitted anywhere)."""

import json
import shutil
from datetime import UTC, datetime

import pytest
from labtestkit import TOY, Clock, FakeKaggle, approve_and_lock, runs

from labcore.io import write_text_atomic
from labcore.kaggle.client import KaggleError
from labcore.lab import Lab
from labcore.runs import dispatch as dispatching
from labcore.runs import state as st

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)  # a Thursday


def go(root, backend, exp="exp_001", **options):
    options.setdefault("username", "tester")
    options.setdefault("now", NOW)
    return dispatching.dispatch(Lab.at(root), TOY, exp, backend, **options)


def add_experiment(root, name, hours):
    configs = root / "directions" / TOY / "configs"
    text = (configs / "exp_001.yaml").read_text().replace("est_gpu_hours: 0.5", f"est_gpu_hours: {hours}")
    (configs / f"{name}.yaml").write_text(text)


def set_cap(root, hours):
    path = root / "directions" / TOY / "direction.yaml"
    text = path.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if "l1_gpu_hours_cap" in line)
    write_text_atomic(path, text.replace(line, f"  l1_gpu_hours_cap: {hours}"))


def set_weekly(root, cap):
    budget = st.default_budget()
    budget["weekly_cap_hours"] = cap
    st.StateStore(root / "state").save_budget(budget)


# ---------------------------------------------------------------- budget arithmetic


@pytest.mark.parametrize("estimate, limit", [(0.5, 1.0), (1.5, 3.0), (6, 11.5), (11, 11.5)])
def test_timeout_is_twice_the_estimate_up_to_the_hard_limit(estimate, limit):
    assert st.timeout_hours(estimate) == limit


@pytest.mark.parametrize(
    "day, expected",
    [(21, "2026-11-21"), (26, "2026-11-21"), (27, "2026-11-21"), (28, "2026-11-28")],
)
def test_week_starts_on_the_reset_day(day, expected):
    assert st.week_start(datetime(2026, 11, day, tzinfo=UTC), "saturday") == expected


def test_a_new_week_keeps_only_unfinished_reservations():
    budget = st.default_budget()
    st.reserve(budget, "old-done", "a", 3, NOW)
    st.reserve(budget, "old-running", "a", 2, NOW)
    st.settle(budget, "old-done", "a", 2.5, "2026-11-26T13:00:00Z", NOW)
    assert st.used_hours(budget) == 4.5
    st.roll_week(budget, datetime(2026, 11, 29, tzinfo=UTC))
    assert [e["run_id"] for e in budget["entries"]] == ["old-running"]
    assert budget["week_start"] == "2026-11-28"


def test_a_run_finished_last_week_does_not_count_this_week():
    budget = st.default_budget()
    st.settle(budget, "r", "a", 5, "2026-11-20T10:00:00Z", NOW)
    assert st.used_hours(budget) == 0


def test_direction_hours_prefers_actual_over_estimate():
    records = [
        {"slug": "a", "est_gpu_hours": 2.0, "actual_gpu_hours": 0.5},
        {"slug": "a", "est_gpu_hours": 2.0, "actual_gpu_hours": None},
        {"slug": "b", "est_gpu_hours": 9.0},
    ]
    assert st.direction_hours(records, "a") == 2.5


def test_corrupt_state_is_refused(tmp_path):
    (tmp_path / "runs.json").write_text(json.dumps({"runs": [{"run_id": "x"}]}))
    with pytest.raises(st.StateError):
        st.StateStore(tmp_path).runs()


# ---------------------------------------------------------------- dispatch


def test_an_unlocked_direction_is_never_pushed_to_kaggle(toy_lab):
    kaggle = FakeKaggle()
    with pytest.raises(dispatching.NotLockedError, match="SCAFFOLDED"):
        go(toy_lab, kaggle)
    assert kaggle.pushes == [] and runs(toy_lab) == []


def test_a_changed_gate_blocks_the_push(toy_lab):
    approve_and_lock(toy_lab, TOY)
    with open(toy_lab / "directions" / TOY / "prereg.md", "a", encoding="utf-8") as handle:
        handle.write("edited\n")
    with pytest.raises(dispatching.NotLockedError, match="changed after the lock"):
        go(toy_lab, FakeKaggle())


def test_submission_is_recorded(toy_lab):
    locked = approve_and_lock(toy_lab, TOY)
    kaggle = FakeKaggle()
    outcome = go(toy_lab, kaggle)
    assert outcome.action == "submitted"
    [record] = runs(toy_lab)
    assert record == outcome.record
    expected = {
        "slug": TOY, "exp": "exp_001", "kernel_id": f"tester/{TOY}-exp-001", "status": "QUEUED",
        "submitted_at": "2026-11-26T12:00:00Z", "est_gpu_hours": 0.5, "retries": 0, "commit_sha": "unknown",
        "prereg_hash": locked["hash"], "l1_round": 1, "backend": "kaggle", "failure_class": None,
        "issue_number": None, "finished_at": None, "last_checked_at": None,
    }
    assert {key: record[key] for key in expected} == expected
    assert record["result_path"] == f"results/{TOY}/exp_001/{record['run_id']}"
    assert record["config_hash"].startswith("sha256:")
    assert kaggle.pushes[0]["timeout_s"] == 3600  # twice the 0.5 h estimate
    budget = st.StateStore(toy_lab / "state").budget()
    assert budget["used_hours"] == 0.5 and budget["entries"][0]["settled"] is False


def test_the_username_comes_from_the_environment(toy_lab, monkeypatch):
    approve_and_lock(toy_lab, TOY)
    with pytest.raises(ValueError, match="KAGGLE_USERNAME"):
        go(toy_lab, FakeKaggle(), username=None)
    monkeypatch.setenv("KAGGLE_USERNAME", "from-env")
    assert go(toy_lab, FakeKaggle(), username=None).record["kernel_id"].startswith("from-env/")


def test_repeated_triggers_do_not_spend_twice(toy_lab):
    approve_and_lock(toy_lab, TOY)
    kaggle = FakeKaggle()
    first = go(toy_lab, kaggle).record
    assert go(toy_lab, kaggle).action == "skipped"  # still active
    with pytest.raises(dispatching.DispatchError, match="still active"):
        go(toy_lab, kaggle, force=True)
    store = st.StateStore(toy_lab / "state")
    store.put_run({**first, "status": "COMPLETE", "actual_gpu_hours": 0.2})
    assert go(toy_lab, kaggle).action == "skipped"  # already done
    assert len(kaggle.pushes) == 1
    assert go(toy_lab, kaggle, force=True).action == "submitted"


def test_direction_cap(toy_lab):
    set_cap(toy_lab, 1)
    add_experiment(toy_lab, "exp_big", 0.8)
    approve_and_lock(toy_lab, TOY)
    kaggle = FakeKaggle()
    go(toy_lab, kaggle)
    with pytest.raises(dispatching.BudgetError, match="l1_gpu_hours_cap"):
        go(toy_lab, kaggle, "exp_big", enqueue=True)
    assert len(kaggle.pushes) == 1 and st.StateStore(toy_lab / "state").queue() == []


def test_weekly_cap_refuses_or_enqueues_and_the_queue_drains(toy_lab):
    add_experiment(toy_lab, "exp_002", 0.8)
    approve_and_lock(toy_lab, TOY)
    set_weekly(toy_lab, 1)
    kaggle, store = FakeKaggle(), st.StateStore(toy_lab / "state")
    go(toy_lab, kaggle)
    with pytest.raises(dispatching.BudgetError, match="weekly"):
        go(toy_lab, kaggle, "exp_002")
    assert store.queue() == []

    assert go(toy_lab, kaggle, "exp_002", enqueue=True).action == "queued"
    assert go(toy_lab, kaggle, "exp_002", enqueue=True).action == "queued"
    assert [(q["slug"], q["exp"]) for q in store.queue()] == [(TOY, "exp_002")]
    assert len(kaggle.pushes) == 1

    report = dispatching.drain_queue(Lab.at(toy_lab), kaggle, username="tester", now=NOW)
    assert "weekly" in report[0] and len(store.queue()) == 1

    later = Clock(datetime(2026, 12, 6, tzinfo=UTC))
    budget = store.budget()
    st.settle(budget, runs(toy_lab)[0]["run_id"], TOY, 0.1, "2026-11-26T13:00:00Z", NOW)
    store.save_budget(budget)
    report = dispatching.drain_queue(Lab.at(toy_lab), kaggle, username="tester", now=later())
    assert "submitted" in report[0]
    assert store.queue() == [] and len(kaggle.pushes) == 2


def test_a_failed_push_changes_nothing(toy_lab):
    approve_and_lock(toy_lab, TOY)

    class Broken(FakeKaggle):
        def push(self, kernel_dir, *, timeout_s=None):
            raise KaggleError("kernel push failed: quota")

    with pytest.raises(KaggleError):
        go(toy_lab, Broken())
    assert runs(toy_lab) == []
    assert st.StateStore(toy_lab / "state").budget()["entries"] == []


def test_dry_run_cli_stays_inside_dryrun(toy_lab, capsys):
    code = dispatching.main([TOY, "exp_001", "--dry-run", "--root", str(toy_lab)])
    assert code == 0, capsys.readouterr()
    assert not (toy_lab / "state" / "runs.json").exists()
    [record] = st.StateStore(toy_lab / ".dryrun" / "state").runs()
    assert record["backend"] == "local" and record["prereg_hash"] is None
    assert "submitted" in capsys.readouterr().out


def test_cli_on_an_unlocked_direction(toy_lab, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setattr(dispatching, "KaggleClient", FakeKaggle)
    arguments = [TOY, "exp_001", "--root", str(toy_lab)]
    assert dispatching.main(arguments) == 1
    assert dispatching.main([*arguments, "--skip-unlocked"]) == 0
    assert runs(toy_lab) == []
    approve_and_lock(toy_lab, TOY)
    assert dispatching.main(arguments) == 0
    assert len(runs(toy_lab)) == 1


def test_code_can_come_from_another_checkout(toy_lab, tmp_path, monkeypatch):
    """dispatch.yml: state lives in the main checkout, the kernel code in the pull request's."""
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setattr(dispatching, "KaggleClient", FakeKaggle)
    branch = tmp_path / "pr-code"
    shutil.copytree(toy_lab, branch)
    approve_and_lock(branch, TOY)  # locked only in the branch
    marker = tmp_path / "executed"
    for module in (branch / "labcore" / "__init__.py", branch / "directions" / TOY / "src" / "main.py"):
        with open(module, "a", encoding="utf-8") as handle:
            handle.write(f"\nopen({str(marker)!r}, 'w').close()\n")

    code = dispatching.main([TOY, "exp_001", "--root", str(toy_lab), "--source-root", str(branch)])
    assert code == 0
    [record] = runs(toy_lab)
    assert record["prereg_hash"] and record["result_path"].startswith("results/")
    assert not (branch / "state" / "runs.json").exists()
    assert not marker.exists()  # bundled, never run on the dispatching machine
    with pytest.raises(SystemExit):
        dispatching.main([TOY, "exp_001", "--dry-run", "--source-root", str(branch)])


def test_cli_reports_unknown_names(toy_lab, capsys):
    assert dispatching.main(["no-such", "exp_001", "--root", str(toy_lab)]) == 1
    assert dispatching.main([TOY, "no_such_exp", "--root", str(toy_lab)]) == 1
    assert "dispatch failed" in capsys.readouterr().err
