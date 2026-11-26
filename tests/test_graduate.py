"""Graduation, in throwaway git repositories."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from labtestkit import TOY, approve_and_lock, commit_all, git, tool

from labcore import registry
from labcore.io import write_json, write_text_atomic

graduate = tool("graduate")
validate = tool("validate")


def history_methods():
    done = subprocess.run(["git", "filter-repo", "--version"], capture_output=True, check=False)
    missing = pytest.mark.skipif(done.returncode != 0, reason="git filter-repo is not installed")
    return ["subtree", pytest.param("filter-repo", marks=missing)]


@pytest.fixture
def go_lab(toy_lab) -> Path:
    """A committed lab whose direction has a GO verdict and two commits of history."""
    commit_all(toy_lab, "scaffold the direction")
    readme = toy_lab / "directions" / TOY / "src" / "notes.txt"
    readme.write_text("second step\n")
    commit_all(toy_lab, "add notes to the direction")
    (toy_lab / "unrelated.txt").write_text("lab only\n")
    commit_all(toy_lab, "unrelated lab change")
    approve_and_lock(toy_lab, TOY)
    write_json(toy_lab / "results" / TOY / "exp_001" / "run-1" / "scorecard.json", {"run_id": "run-1"})
    registry.set_status(toy_lab, TOY, "VERDICT", verdict="GO")
    commit_all(toy_lab, "record the verdict")
    return toy_lab


def clean_python(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTEST_ADDOPTS")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("method", history_methods())
def test_graduated_folder_stands_alone(go_lab, tmp_path, method):
    out = tmp_path / "out" / TOY
    refs_before = git(go_lab, "for-each-ref")
    head_before = git(go_lab, "rev-parse", "HEAD")

    result = graduate.graduate(go_lab, TOY, out, method)
    assert result["method"] == method and result["source_commit"] == head_before

    for name in (
        "direction.yaml", "prereg.md", "VERDICT.md", "checks.py", "requirements.txt", "src/main.py",
        "src/notes.txt", "configs/exp_001.yaml", "tests/test_l0_smoke.py", "README.md", "VENDORED.md",
        "pyproject.toml", ".gitignore", ".gitattributes", "labcore/__init__.py",
        "labcore/schemas/scorecard.schema.json", "evidence/exp_001/run-1/scorecard.json",
        "evidence/registry_entry.yaml",
    ):
        assert (out / name).is_file(), name
    assert not (out / "directions").exists() and not (out / "unrelated.txt").exists()
    assert not (out / "tools").exists()
    assert "status: GRADUATED" in (out / "direction.yaml").read_text(encoding="utf-8")
    assert head_before in (out / "VENDORED.md").read_text()

    # history of the direction only, no remote, nothing committed by the tool
    subjects = git(out, "log", "--format=%s").splitlines()
    assert {"scaffold the direction", "add notes to the direction"} <= set(subjects)
    assert "unrelated lab change" not in subjects
    assert git(out, "branch", "--show-current") == "main"
    assert git(out, "remote") == ""
    assert "labcore/" in git(out, "status", "--porcelain")
    assert git(go_lab, "rev-parse", "HEAD") == head_before
    assert git(go_lab, "for-each-ref") == refs_before  # no temporary branch left behind

    item = registry.load(go_lab / "registry.yaml")["directions"][TOY]
    assert (item["status"], item["graduated_to"]) == ("GRADUATED", TOY)
    assert validate.validate_lab(go_lab) == []
    assert tool("prereg_lock").check(go_lab, "HEAD") == []

    # With the lab gone, the folder's own tests pass and labcore is the vendored copy.
    graduate._remove_tree(go_lab)
    where = clean_python(out, "-c", "import labcore; print(labcore.__file__)")
    assert Path(where.stdout.strip()).resolve().is_relative_to(out.resolve()), where.stderr
    tests = clean_python(out, "-m", "pytest", "-q", "-p", "no:cacheprovider")
    assert tests.returncode == 0, tests.stdout + tests.stderr
    assert "1 passed" in tests.stdout
    assert graduate.lab_imports(out) == []


@pytest.mark.parametrize(
    "status, verdict",
    [("VERDICT", "REVISE"), ("VERDICT", "PARK"), ("L1_RUNNING", None), ("ARCHIVED", "GO")],
)
def test_only_go_graduates(go_lab, tmp_path, status, verdict):
    registry.set_status(go_lab, TOY, status, verdict=verdict)
    commit_all(go_lab, "other outcome")
    with pytest.raises(graduate.GraduationError, match="only a direction with a GO verdict"):
        graduate.graduate(go_lab, TOY, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_changed_gates_block_graduation(go_lab, tmp_path):
    with open(go_lab / "directions" / TOY / "prereg.md", "a", encoding="utf-8") as handle:
        handle.write("later edit\n")
    commit_all(go_lab, "edit")
    with pytest.raises(graduate.GraduationError, match="differs from the lock"):
        graduate.graduate(go_lab, TOY, tmp_path / "out")


def test_uncommitted_work_blocks_graduation(go_lab, tmp_path):
    (go_lab / "directions" / TOY / "src" / "new.py").write_text("x = 1\n")
    with pytest.raises(graduate.GraduationError, match="commit these changes"):
        graduate.graduate(go_lab, TOY, tmp_path / "out")


def test_a_lab_without_its_own_repository_is_refused(toy_lab, tmp_path):
    approve_and_lock(toy_lab, TOY)
    registry.set_status(toy_lab, TOY, "VERDICT", verdict="GO")
    with pytest.raises(graduate.GraduationError, match="git repository"):
        graduate.graduate(toy_lab, TOY, tmp_path / "out")


def test_destination_rules(go_lab, tmp_path):
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("precious\n")
    for destination in (busy, go_lab / "inside", go_lab, tmp_path):
        with pytest.raises(graduate.GraduationError):
            graduate.graduate(go_lab, TOY, destination)
    assert (busy / "keep.txt").read_text() == "precious\n"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert graduate.graduate(go_lab, TOY, empty, "subtree")["out"] == empty.resolve()


def test_lab_only_imports_are_refused(go_lab, tmp_path):
    module = go_lab / "directions" / TOY / "src" / "main.py"
    text = module.read_text(encoding="utf-8")
    write_text_atomic(module, text + "\n\ndef helper():\n    from tools import validate\n")
    commit_all(go_lab, "bad import")
    with pytest.raises(graduate.GraduationError, match="imports tools"):
        graduate.graduate(go_lab, TOY, tmp_path / "out")


def test_a_failure_leaves_nothing_behind(go_lab, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(graduate, "_copy_evidence", broken)
    with pytest.raises(OSError):
        graduate.graduate(go_lab, TOY, tmp_path / "out", "subtree")
    assert not (tmp_path / "out").exists()
    assert registry.load(go_lab / "registry.yaml")["directions"][TOY]["status"] == "VERDICT"
    shutil.rmtree(tmp_path / "out", ignore_errors=True)


def test_cli_prints_the_commands_and_runs_none_of_them(go_lab, tmp_path, capsys):
    out = tmp_path / "standalone"
    assert graduate.main([TOY, "--out", str(out), "--root", str(go_lab)]) == 0
    printed = capsys.readouterr().out
    assert f"gh repo create {TOY} --private" in printed and "--push" in printed
    assert git(out, "remote") == ""
    assert graduate.main([TOY, "--out", str(out), "--root", str(go_lab)]) == 1
