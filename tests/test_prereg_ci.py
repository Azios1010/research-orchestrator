"""The pre-registration lock, the lock tool and the CI scripts, on throwaway git repos."""

import re

import pytest
from labtestkit import (
    TOY,
    FakeKaggle,
    approve_and_lock,
    commit_all,
    git,
    mark_l0,
    set_gate_approval,
    tool,
)

import labcore
from labcore import registry
from labcore.io import write_text_atomic
from labcore.lab import Lab
from labcore.runs import dispatch as dispatching
from labcore.verdict.prereg import prereg_hash, stats_version

lock_tool = tool("lock_prereg")
lock_check = tool("prereg_lock")
protected = tool("protected_paths")
changed = tool("changed")


def edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    write_text_atomic(path, text.replace(old, new, 1))


def entry(root):
    return registry.load(root / "registry.yaml")["directions"][TOY]


def bump_version(root):
    path = root / "labcore" / "__init__.py"
    text, count = re.subn(r'__version__ = "[^"]+"', '__version__ = "99.0.0"', path.read_text())
    assert count == 1
    write_text_atomic(path, text)


LOCKED_EDITS = {
    "prereg.md": ("prereg.md", "## Question", "## Questions"),
    "gates": ("direction.yaml", 'GO: "ci_lo(a-b) > 0"', 'GO: "ci_lo(a-b) > -0.01"'),
    "scorecard_map": ("direction.yaml", "path: contrasts.a-b.ci_lo}", "path: contrasts.a-b.mean_delta}"),
    "checks.py": ("checks.py", "usable * 5 < requested * 4", "usable * 5 < requested * 1"),
    "evaluate.py": ("src/evaluate.py", 'scores["a"].mean()', 'scores["a"].median()'),
    "labcore.__version__": None,
}


def apply_locked_edit(root, part):
    if LOCKED_EDITS[part] is None:
        bump_version(root)
    else:
        name, old, new = LOCKED_EDITS[part]
        edit(root / "directions" / TOY / name, old, new)


@pytest.mark.parametrize("part", LOCKED_EDITS)
def test_every_locked_part_changes_the_hash_and_is_refused(toy_lab, part):
    approve_and_lock(toy_lab, TOY)
    directory = toy_lab / "directions" / TOY
    before = prereg_hash(directory)
    assert lock_check.check(toy_lab) == []

    apply_locked_edit(toy_lab, part)

    assert prereg_hash(directory) != before
    [error] = lock_check.check(toy_lab)
    assert "changed after the lock" in error
    assert lock_check.main(["--root", str(toy_lab)]) == 1
    assert any("differs from the lock" in e for e in tool("validate").validate_lab(toy_lab))
    kaggle = FakeKaggle()
    with pytest.raises(dispatching.NotLockedError, match="changed after the lock"):
        dispatching.dispatch(Lab.at(toy_lab), TOY, "exp_001", kaggle, username="tester")
    assert kaggle.pushes == []
    verdict_tool = tool("verdict")
    with pytest.raises(verdict_tool.VerdictError, match="differs from the lock"):
        verdict_tool.record_verdict(Lab.at(toy_lab), TOY)


def test_hash_ignores_what_is_not_locked(toy_lab):
    directory = toy_lab / "directions" / TOY
    original = prereg_hash(directory)

    for name in ("prereg.md", "checks.py", "src/evaluate.py"):  # Windows line endings
        path = directory / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    edit(directory / "direction.yaml", "# DRAFT: pending", "# reworded comment: pending")
    edit(directory / "direction.yaml", 'owner_notes: ""', 'owner_notes: "note"')
    registry.set_status(toy_lab, TOY, "L0_PASS")
    with open(directory / "src" / "main.py", "a", encoding="utf-8") as handle:
        handle.write("# main.py is not covered\n")
    with open(toy_lab / "labcore" / "io.py", "a", encoding="utf-8") as handle:
        handle.write("# labcore counts only through its version\n")

    assert prereg_hash(directory) == original


def test_the_version_is_read_without_running_the_checkout(toy_lab):
    init = toy_lab / "labcore" / "__init__.py"
    write_text_atomic(init, '__version__ = "7.1"\nraise SystemExit("executed")\n')
    assert stats_version(toy_lab) == "7.1"
    write_text_atomic(init, "__version__ = compute()\n")
    with pytest.raises(ValueError, match="literal"):
        prereg_hash(toy_lab / "directions" / TOY)


@pytest.mark.parametrize("name", ["src/evaluate.py", "checks.py"])
def test_a_missing_locked_file(toy_lab, name):
    directory = toy_lab / "directions" / TOY
    set_gate_approval(toy_lab, TOY)
    (directory / name).unlink()
    mark_l0(toy_lab, TOY)
    with pytest.raises(lock_tool.LockError, match=name):
        lock_tool.lock(toy_lab, TOY)
    assert any(f"missing {name}" in e for e in tool("validate").validate_lab(toy_lab))


@pytest.mark.parametrize("name", ["src/evaluate.py", "checks.py"])
def test_deleting_a_locked_file_is_caught(toy_lab, name):
    approve_and_lock(toy_lab, TOY)
    (toy_lab / "directions" / TOY / name).unlink()
    assert "cannot hash" in lock_check.check(toy_lab)[0]
    with pytest.raises(dispatching.NotLockedError, match="cannot be checked"):
        dispatching.dispatch(Lab.at(toy_lab), TOY, "exp_001", FakeKaggle(), username="tester")


def test_a_finished_direction_survives_a_version_bump_but_not_a_code_change(toy_lab):
    locked = approve_and_lock(toy_lab, TOY)
    assert locked["stats_version"] == labcore.__version__
    registry.set_status(toy_lab, TOY, "VERDICT", verdict="GO")
    bump_version(toy_lab)
    assert lock_check.check(toy_lab) == []
    assert tool("validate").validate_lab(toy_lab) == []
    apply_locked_edit(toy_lab, "evaluate.py")
    assert "changed after the lock" in lock_check.check(toy_lab)[0]


def test_bumping_the_version_while_recording_a_verdict_is_caught(locked_repo):
    bump_version(locked_repo)
    registry.set_status(locked_repo, TOY, "VERDICT", verdict="GO")
    commit_all(locked_repo, "verdict and bump together")
    assert lock_check.check(locked_repo) == []  # looks finished ...
    errors = lock_check.check(locked_repo, "main")  # ... but was running on the base branch
    assert len(errors) == 1 and "changed after the lock" in errors[0]


def test_a_version_bump_after_the_verdict_was_merged_is_fine(locked_repo):
    registry.set_status(locked_repo, TOY, "VERDICT", verdict="GO")
    commit_all(locked_repo, "verdict")
    git(locked_repo, "branch", "-f", "main", "HEAD")
    bump_version(locked_repo)
    commit_all(locked_repo, "labcore release")
    assert lock_check.check(locked_repo, "main") == []


def test_lock_needs_l0_approval_and_unchanged_code(toy_lab):
    with pytest.raises(lock_tool.LockError, match="L0_PASS"):
        lock_tool.lock(toy_lab, TOY)
    mark_l0(toy_lab, TOY)
    with pytest.raises(lock_tool.LockError, match="DRAFT"):
        lock_tool.lock(toy_lab, TOY)
    set_gate_approval(toy_lab, TOY)
    with open(toy_lab / "directions" / TOY / "src" / "main.py", "a", encoding="utf-8") as handle:
        handle.write("# changed after L0\n")
    with pytest.raises(lock_tool.LockError, match="code changed"):
        lock_tool.lock(toy_lab, TOY)
    mark_l0(toy_lab, TOY)

    locked = lock_tool.lock(toy_lab, TOY)
    assert locked["round"] == 1
    assert entry(toy_lab)["status"] == "L1_RUNNING"
    assert registry.read_status_from_yaml(toy_lab / "directions" / TOY) == "L1_RUNNING"
    assert lock_check.check(toy_lab) == []
    assert tool("validate").validate_lab(toy_lab) == []


@pytest.fixture
def locked_repo(toy_lab):
    approve_and_lock(toy_lab, TOY)
    commit_all(toy_lab, "locked")
    git(toy_lab, "checkout", "-q", "-b", "pr")
    return toy_lab


def test_unrelated_change_passes_both_checks(locked_repo):
    (locked_repo / "notes.md").write_text("hello\n")
    commit_all(locked_repo, "notes")
    assert lock_check.check(locked_repo, "main") == []
    assert protected.violations(locked_repo, "main") == []
    assert protected.main(["--base", "main", "--root", str(locked_repo)]) == 0


def test_relocking_edited_gates_is_caught_against_the_base_branch(locked_repo):
    directory = locked_repo / "directions" / TOY
    edit(directory / "direction.yaml", 'GO: "ci_lo(a-b) > 0"', 'GO: "ci_lo(a-b) > -1"')
    document = registry.load(locked_repo / "registry.yaml")
    document["directions"][TOY]["prereg_lock"]["hash"] = prereg_hash(directory)
    registry.save(locked_repo / "registry.yaml", document)
    commit_all(locked_repo, "move the goalposts")

    assert lock_check.check(locked_repo) == []  # self-consistent ...
    errors = lock_check.check(locked_repo, "main")  # ... but not the lock that was merged
    assert any("was changed or removed" in error for error in errors)


@pytest.mark.parametrize("status", ["SCAFFOLDED", "L0_PASS", "ARCHIVED"])
def test_moving_the_status_back_is_caught(locked_repo, status):
    document = registry.load(locked_repo / "registry.yaml")
    document["directions"][TOY].update(status=status, prereg_lock=None)
    registry.save(locked_repo / "registry.yaml", document)
    commit_all(locked_repo, "step back")
    assert lock_check.check(locked_repo, "main")


def test_removing_a_locked_direction_is_caught(locked_repo):
    registry.save(locked_repo / "registry.yaml", {"directions": {}})
    commit_all(locked_repo, "remove")
    assert "removed" in lock_check.check(locked_repo, "main")[0]


def test_new_round_after_revise_is_allowed_but_limited(toy_lab):
    approve_and_lock(toy_lab, TOY)
    with pytest.raises(lock_tool.LockError, match="REVISE"):
        lock_tool.new_round(toy_lab, TOY)
    registry.set_status(toy_lab, TOY, "VERDICT", verdict="REVISE")
    commit_all(toy_lab, "revise verdict")
    git(toy_lab, "checkout", "-q", "-b", "round-2")

    lock_tool.new_round(toy_lab, TOY)
    item = entry(toy_lab)
    assert item["status"] == "SCAFFOLDED" and item["l1_rounds"] == 1
    assert item["lock_history"][0]["verdict"] == "REVISE"
    assert "approval: DRAFT" in (toy_lab / "directions" / TOY / "direction.yaml").read_text()
    with open(toy_lab / "directions" / TOY / "prereg.md", "a", encoding="utf-8") as handle:
        handle.write("Round 2 change.\n")
    commit_all(toy_lab, "open round 2")
    assert lock_check.check(toy_lab, "main") == []

    assert approve_and_lock(toy_lab, TOY)["round"] == 2
    commit_all(toy_lab, "lock round 2")
    assert lock_check.check(toy_lab, "main") == []
    registry.set_status(toy_lab, TOY, "VERDICT", verdict="REVISE")
    with pytest.raises(lock_tool.LockError, match="rounds are used"):
        lock_tool.new_round(toy_lab, TOY)


def test_a_go_verdict_cannot_open_a_new_round(locked_repo):
    registry.set_status(locked_repo, TOY, "VERDICT", verdict="GO")
    commit_all(locked_repo, "go")
    git(locked_repo, "branch", "-f", "main", "HEAD")
    document = registry.load(locked_repo / "registry.yaml")
    item = document["directions"][TOY]
    history = [{**item["prereg_lock"], "verdict": "GO"}]
    item.update(status="SCAFFOLDED", lock_history=history, prereg_lock=None)
    registry.save(locked_repo / "registry.yaml", document)
    commit_all(locked_repo, "fake new round")
    assert lock_check.check(locked_repo, "main")


@pytest.mark.parametrize("path", ["state/runs.json", "results/x/y/z.json"])
def test_protected_paths(locked_repo, path):
    target = locked_repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n")
    commit_all(locked_repo, "touch bot files")
    assert protected.violations(locked_repo, "main") == [path]
    assert protected.main(["--base", "main", "--root", str(locked_repo)]) == 1


def test_moving_a_file_out_of_results_is_still_a_violation(locked_repo):
    target = locked_repo / "results" / "keep.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("result\n")
    commit_all(locked_repo, "bot")
    git(locked_repo, "branch", "-f", "main", "HEAD")
    git(locked_repo, "mv", "results/keep.txt", "keep.txt")
    commit_all(locked_repo, "move")
    assert protected.violations(locked_repo, "main") == ["results/keep.txt"]


def test_bot_commits_on_main_do_not_count_against_a_branch(locked_repo):
    git(locked_repo, "checkout", "-q", "main")
    (locked_repo / "state" / "runs.json").write_text('{"runs": []}\n')
    commit_all(locked_repo, "bot commit on main")
    git(locked_repo, "checkout", "-q", "pr")
    (locked_repo / "notes.md").write_text("x\n")
    commit_all(locked_repo, "pr work")
    assert protected.violations(locked_repo, "main") == []


def test_unknown_base_ref_fails_loudly(locked_repo):
    assert lock_check.main(["--base", "no-such-ref", "--root", str(locked_repo)]) == 2
    assert protected.main(["--base", "no-such-ref", "--root", str(locked_repo)]) == 2


def test_changed_directions_and_configs(locked_repo):
    assert changed.changed_directions(locked_repo, "main") == []
    config = locked_repo / "directions" / TOY / "configs" / "exp_002.yaml"
    config.write_text("compute: {est_gpu_hours: 1}\nparams: {}\n")
    template = locked_repo / "directions" / "_template" / "configs" / "exp_001.yaml"
    template.write_text(template.read_text() + "# touched\n")
    commit_all(locked_repo, "second experiment")
    assert changed.changed_directions(locked_repo, "main") == [TOY]
    assert changed.changed_configs(locked_repo, "main") == [{"slug": TOY, "exp": "exp_002"}]


@pytest.mark.parametrize("change", ["edit", "delete", "add"])
def test_a_labcore_change_selects_every_direction(toy_lab, change):
    tool("new_direction").new_direction(toy_lab, "other-direction", "second")
    commit_all(toy_lab, "two directions")
    git(toy_lab, "checkout", "-q", "-b", "pr")
    (toy_lab / "docs" / "labcore").mkdir(parents=True)
    (toy_lab / "docs" / "labcore" / "note.md").write_text("not the library\n")
    commit_all(toy_lab, "docs only")
    assert changed.changed_directions(toy_lab, "main") == []

    if change == "edit":
        with open(toy_lab / "labcore" / "io.py", "a", encoding="utf-8") as handle:
            handle.write("# touched\n")
    elif change == "delete":
        (toy_lab / "labcore" / "notify" / "telegram.py").unlink()
    else:
        (toy_lab / "labcore" / "new_module.py").write_text("x = 1\n")
    commit_all(toy_lab, "library change")
    assert changed.changed_directions(toy_lab, "main") == ["other-direction", TOY]  # not _template


def test_a_registry_only_change_selects_the_direction(locked_repo):
    registry.set_status(locked_repo, TOY, "VERDICT", verdict="PARK")
    commit_all(locked_repo, "verdict")
    assert changed.changed_directions(locked_repo, "main") == [TOY]
