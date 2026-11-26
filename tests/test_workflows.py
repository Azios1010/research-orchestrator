"""The workflows: structure, safety rules, and their shell scripts run locally against stubs."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from labtestkit import REPO, TOY, commit_all, git

from labcore.runs.state import ACTIVE_STATUSES

FOLDER = REPO / ".github" / "workflows"
NAMES = ["ci-l0", "claude", "dispatch", "watcher"]


def load(name):
    return yaml.safe_load((FOLDER / f"{name}.yml").read_text(encoding="utf-8"))


def triggers(name):
    document = load(name)
    return document.get("on", document.get(True))  # YAML 1.1 reads a bare "on" as true


def steps(name):
    return [step for job in load(name)["jobs"].values() for step in job.get("steps", [])]


def step(name, title):
    [found] = [s for s in steps(name) if s.get("name") == title]
    return found


def test_the_four_workflows_exist_and_parse():
    assert sorted(p.stem for p in FOLDER.glob("*.yml")) == NAMES
    for name in NAMES:
        assert load(name)["jobs"] and triggers(name)


@pytest.mark.parametrize("name", NAMES)
def test_actions_are_pinned_and_permissions_declared(name):
    document = load(name)
    assert document["permissions"] == {"contents": "read"}
    for used in (s["uses"] for s in steps(name) if "uses" in s):
        if used == "anthropics/claude-code-action@v1":  # the major tag the setup asked for
            continue
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", used), used
    assert "concurrency" in document or any("concurrency" in job for job in document["jobs"].values())


@pytest.mark.parametrize("name", NAMES)
def test_no_expression_is_pasted_into_a_script(name):
    """User-controlled values reach scripts through env, never by text substitution."""
    for item in steps(name):
        assert "${{" not in item.get("run", ""), item.get("name")


def test_ci_has_no_secrets_and_no_write_access():
    text = (FOLDER / "ci-l0.yml").read_text(encoding="utf-8")
    assert "secrets." not in text and ": write" not in text
    assert {"pull_request", "workflow_call"} <= set(triggers("ci-l0"))
    runs = " ".join(s.get("run", "") for s in steps("ci-l0"))
    for needed in (
        "ruff check", "tools/validate.py", "tools/ci/protected_paths.py", "tools/ci/prereg_lock.py",
        "pytest tests", "tools/ci/changed.py directions", "--run-l0", "tools/ci/results_privacy.py",
    ):
        assert needed in runs, needed


def test_only_one_job_at_a_time_touches_state():
    for name, job in (("dispatch", "dispatch"), ("watcher", "watch")):
        concurrency = load(name)["jobs"][job]["concurrency"]
        assert concurrency == {"group": "lab-state", "cancel-in-progress": False, "queue": "max"}


def test_triggers():
    dispatch = triggers("dispatch")
    assert dispatch["push"] == {"branches": ["main"], "paths": ["directions/*/configs/**"]}
    assert dispatch["pull_request"] == {"types": ["labeled"]}
    assert {"slug", "exp"} <= set(dispatch["workflow_dispatch"]["inputs"])
    assert {"schedule", "workflow_dispatch"} <= set(triggers("watcher"))
    assert set(triggers("claude")) == {"issue_comment", "pull_request_review_comment"}
    assert "minutes" in (FOLDER / "watcher.yml").read_text(encoding="utf-8").split("name:")[0].lower()


def test_labelled_pull_requests_are_guarded():
    jobs = load("dispatch")["jobs"]
    guard = jobs["ci"]["if"]
    for condition in (
        "github.event.label.name == 'run-l1'",
        "github.actor == github.repository_owner",
        "github.event.pull_request.head.repo.full_name == github.repository",
    ):
        assert condition in guard
    assert jobs["ci"]["uses"] == "./.github/workflows/ci-l0.yml"
    assert jobs["dispatch"]["needs"] == "ci"
    checkout = step("dispatch", "Check out the pull request (read only, never executed here)")
    assert checkout["with"]["persist-credentials"] is False and checkout["with"]["path"] == "pr-code"


def test_claude_is_owner_only_and_keeps_comment_mode():
    job = load("claude")["jobs"]["claude"]
    assert "github.actor == github.repository_owner" in job["if"] and "@claude" in job["if"]
    assert job["permissions"] == {
        "contents": "write", "pull-requests": "write", "issues": "write", "id-token": "write",
    }
    action = steps("claude")[-1]["with"]
    assert "prompt" not in action and "CLAUDE.md" in action["claude_args"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" in action["claude_code_oauth_token"]
    assert "ANTHROPIC_API_KEY" in action["anthropic_api_key"]


def test_actionlint(tmp_path):
    executable = shutil.which("actionlint")
    if not executable:
        pytest.skip("actionlint is not installed")
    # actionlint 1.7.12 predates concurrency.queue (documented by GitHub).
    done = subprocess.run(
        [executable, "-no-color", "-ignore", 'unexpected key "queue" for "concurrency" section',
         *map(str, sorted(FOLDER.glob("*.yml")))],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stdout


# ---------------------------------------------------------------- the scripts themselves


def find_bash() -> str | None:
    if sys.platform != "win32":
        return shutil.which("bash")
    # Git for Windows, not the WSL launcher in System32.
    done = subprocess.run(["git", "--exec-path"], capture_output=True, text=True, check=False)
    candidate = Path(done.stdout.strip()).parents[2] / "bin" / "bash.exe"
    return str(candidate) if candidate.is_file() else None


@pytest.fixture
def shell(tmp_path):
    """Run a workflow script the way Actions does (bash -e), with a stub `python` that records
    dispatch calls and forwards everything else to the real interpreter."""
    bash = find_bash()
    if not bash:
        pytest.skip("bash is not available")
    stubs, log = tmp_path / "stubs", tmp_path / "dispatch.log"
    stubs.mkdir()
    real = Path(sys.executable).as_posix()
    stub = (
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-m" ] && [ "$2" = "labcore.runs.dispatch" ]; then\n'
        "  shift 2\n"
        f"  printf '%s|' \"$@\" >> '{log.as_posix()}'; echo >> '{log.as_posix()}'\n"
        '  case "$*" in *failing*) exit 1 ;; esac\n'
        "  exit 0\n"
        "fi\n"
        f"'{real}' \"$@\" | tr -d '\\r'\n"
        'exit "${PIPESTATUS[0]}"\n'
    )
    for name in ("python", "python3"):
        (stubs / name).write_text(stub, encoding="utf-8", newline="\n")
        (stubs / name).chmod(0o755)

    def run(script: str, cwd: Path, **env):
        path = tmp_path / "step.sh"
        # PATH is set inside the script so that it also works with Windows-style folders.
        prefix = f'export PATH="$(cd \'{stubs.as_posix()}\' && pwd):$PATH"\n'
        path.write_text(prefix + script, encoding="utf-8", newline="\n")
        done = subprocess.run(
            [bash, "--noprofile", "--norc", "-e", path.as_posix()], cwd=cwd,
            env={**os.environ, **env}, capture_output=True, text=True, check=False,
        )
        calls = []
        if log.exists():
            calls = [line.rstrip("|").split("|") for line in log.read_text().splitlines()]
            log.unlink()
        return done, calls

    return run


def dispatch_env(event, **values):
    base = {"BEFORE": "", "INPUT_SLUG": "", "INPUT_EXP": "", "INPUT_FORCE": "false", "INPUT_DRAIN": "false"}
    return {**base, "EVENT": event, **values}


def add_config(root, name):
    path = root / "directions" / TOY / "configs" / f"{name}.yaml"
    path.write_text("compute: {est_gpu_hours: 0.2}\nparams: {}\n")


def test_dispatch_on_push(shell, toy_lab):
    script = step("dispatch", "Dispatch")["run"]
    before = commit_all(toy_lab, "base")
    done, calls = shell(script, toy_lab, **dispatch_env("push", BEFORE=before))
    assert done.returncode == 0 and calls == [], done.stderr

    add_config(toy_lab, "exp_failing")
    add_config(toy_lab, "exp_002")
    commit_all(toy_lab, "two experiments")
    done, calls = shell(script, toy_lab, **dispatch_env("push", BEFORE=before))
    assert done.returncode == 1, done.stderr  # one failed ...
    prefix = ["--root", ".", "--enqueue", "--skip-unlocked", TOY]
    assert calls == [[*prefix, "exp_002"], [*prefix, "exp_failing"]]  # ... and both were tried

    done, calls = shell(script, toy_lab, **dispatch_env("push", BEFORE="0" * 40))
    assert done.returncode == 0 and calls == []


def test_dispatch_on_a_labelled_pull_request(shell, toy_lab):
    script = step("dispatch", "Dispatch")["run"]
    branch = toy_lab / "pr-code"
    shutil.copytree(toy_lab, branch)
    base = commit_all(branch, "main")
    git(branch, "update-ref", "refs/remotes/origin/main", base)
    done, calls = shell(script, toy_lab, **dispatch_env("pull_request"))
    assert done.returncode == 1 and calls == [] and "changes no experiment config" in done.stdout

    add_config(branch, "exp_002")
    commit_all(branch, "pr")
    done, calls = shell(script, toy_lab, **dispatch_env("pull_request"))
    assert done.returncode == 0, done.stderr
    assert calls == [["--root", ".", "--enqueue", "--source-root", "pr-code", TOY, "exp_002"]]


@pytest.mark.parametrize(
    "inputs, expected",
    [
        ({"INPUT_SLUG": "a-b", "INPUT_EXP": "e1"}, ["--root", ".", "--enqueue", "--", "a-b", "e1"]),
        (
            {"INPUT_SLUG": "a-b", "INPUT_EXP": "e1", "INPUT_FORCE": "true"},
            ["--root", ".", "--enqueue", "--force", "--", "a-b", "e1"],
        ),
        ({"INPUT_DRAIN": "true"}, ["--root", ".", "--drain-queue"]),
        # hostile input stays one argument
        (
            {"INPUT_SLUG": "x; touch pwned", "INPUT_EXP": "$(touch pwned)"},
            ["--root", ".", "--enqueue", "--", "x; touch pwned", "$(touch pwned)"],
        ),
    ],
)
def test_manual_dispatch(shell, tmp_path, inputs, expected):
    script = step("dispatch", "Dispatch")["run"]
    done, calls = shell(script, tmp_path, **dispatch_env("workflow_dispatch", **inputs))
    assert done.returncode == 0, done.stderr
    assert calls == [expected]
    assert not (tmp_path / "pwned").exists()


def test_manual_dispatch_needs_both_names(shell, tmp_path):
    script = step("dispatch", "Dispatch")["run"]
    done, calls = shell(script, tmp_path, **dispatch_env("workflow_dispatch", INPUT_SLUG="a-b"))
    assert done.returncode == 1 and calls == []


@pytest.mark.parametrize(
    "statuses, active",
    [([], "false"), (["COMPLETE", "ERROR"], "false"), *[([s], "true") for s in ACTIVE_STATUSES]],
)
def test_watcher_idle_check(shell, tmp_path, statuses, active):
    script = step("watcher", "Any active run?")["run"]
    (tmp_path / "state").mkdir()
    runs = ", ".join(f'{{"status": "{s}"}}' for s in statuses)
    (tmp_path / "state" / "runs.json").write_text(f'{{"runs": [{runs}]}}')
    output = tmp_path / "github_output"
    done, _ = shell(script, tmp_path, GITHUB_OUTPUT=str(output))
    assert done.returncode == 0, done.stderr
    assert output.read_text().strip() == f"active={active}"


def test_every_later_watcher_step_is_skipped_when_idle():
    all_steps = steps("watcher")
    start = [s.get("name") for s in all_steps].index("Any active run?")
    for item in all_steps[start + 1 :]:
        assert "steps.active.outputs.active == 'true'" in item["if"], item.get("name")


@pytest.mark.parametrize(
    "workflow, title, committed",
    [
        ("dispatch", "Commit state/", ["state/runs.json"]),
        ("watcher", "Commit state/ and results/", ["results/a/r.json", "state/runs.json"]),
    ],
)
def test_bot_commits(shell, tmp_path, workflow, title, committed):
    """Only the bot folders are committed, and a push that lost a race is rebased."""
    origin, work, other = tmp_path / "origin.git", tmp_path / "work", tmp_path / "other"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    for folder in ("state", "results"):
        (work / folder).mkdir()
        (work / folder / ".gitkeep").write_text("")
    (work / "code.py").write_text("x = 1\n")
    commit_all(work, "start")
    git(work, "push", "-q", "origin", "HEAD:main")
    subprocess.run(["git", "clone", "-q", str(origin), str(other)], check=True, capture_output=True)
    (other / "merged.md").write_text("a pull request merged meanwhile\n")
    commit_all(other, "merge")
    git(other, "push", "-q", "origin", "HEAD:main")

    (work / "state" / "runs.json").write_text('{"runs": []}\n')
    (work / "results" / "a").mkdir()
    (work / "results" / "a" / "r.json").write_text("{}\n")
    (work / "code.py").write_text("x = 2  # must not be committed\n")
    done, _ = shell(step(workflow, title)["run"], work)
    assert done.returncode == 0, done.stderr + done.stdout
    log = git(origin, "log", "--format=%an|%s", "main").splitlines()
    assert log[0].startswith("github-actions[bot]|bot(") and log[1].endswith("|merge")
    assert sorted(git(origin, "show", "--name-only", "--format=", "main").split()) == committed

    done, _ = shell(step(workflow, title)["run"], work)  # nothing new: no commit
    assert done.returncode == 0 and len(git(origin, "log", "--format=%h", "main").splitlines()) == len(log)
