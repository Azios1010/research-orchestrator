"""The kernel bundle, its metadata, the Kaggle CLI wrapper (with a fake runner) and the local backend."""

import base64
import io
import json
import subprocess
import tarfile

import pytest
from labtestkit import REPO, TOY, replace_pipeline

from labcore.kaggle import entrypoint_template
from labcore.kaggle.bundle import MAX_SCRIPT_BYTES, BundleError, BundleTooLargeError, build_kernel, code_hash
from labcore.kaggle.client import (
    KaggleClient,
    KaggleError,
    KernelStatus,
    log_to_text,
    parse_push,
    parse_status,
)
from labcore.kaggle.local_backend import LocalBackend
from labcore.kaggle.metadata import build_metadata, kernel_slug, write_metadata


def build(root, out, **options):
    directory = root / "directions" / TOY
    options.setdefault("config_path", directory / "configs" / "exp_001.yaml")
    return build_kernel(
        out, direction_dir=directory, run_id="r1", exp="exp_001", labcore_dir=root / "labcore", **options
    )


def unpacked_names(script: str) -> list[str]:
    namespace: dict = {}
    start = script.index("PAYLOAD_B64 = (")
    exec(script[start : script.index("\n)\n", start) + 3], namespace)  # noqa: S102 - our own output
    raw = base64.b64decode(namespace["PAYLOAD_B64"])
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        return archive.getnames()


# ---------------------------------------------------------------- bundle


def test_bundle_contents(toy_lab, tmp_path):
    directory = toy_lab / "directions" / TOY
    (directory / "configs" / "exp_002.yaml").write_text("compute: {est_gpu_hours: 1}\nparams: {}\n")
    for name in ("notebooks/analysis.ipynb", "results/old.json", "src/weights.pt", "src/.env"):
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text("should not travel\n")

    info = build(toy_lab, tmp_path / "k")
    script = info.script_path.read_text(encoding="utf-8")
    compile(script, "kernel.py", "exec")
    names = unpacked_names(script)

    assert names == sorted(names) == info.files
    assert "labcore/stats/paired.py" in names
    assert "labcore/schemas/scorecard.schema.json" in names
    assert f"directions/{TOY}/src/main.py" in names
    assert [n for n in names if "/configs/" in n] == [f"directions/{TOY}/configs/exp_001.yaml"]
    for forbidden in ("notebooks/", "/tests/", "results/", "weights.pt", ".env", "VERDICT.md"):
        assert not [n for n in names if forbidden in n], forbidden
    assert any("weights.pt" in warning for warning in info.warnings)
    assert info.size_bytes == info.script_path.stat().st_size <= MAX_SCRIPT_BYTES


def test_bundle_is_deterministic_and_ignores_line_endings(toy_lab, tmp_path):
    first = build(toy_lab, tmp_path / "a")
    module = toy_lab / "directions" / TOY / "src" / "main.py"
    before = code_hash(module.parents[1])
    module.write_bytes(module.read_bytes().replace(b"\n", b"\r\n"))
    second = build(toy_lab, tmp_path / "b")
    assert first.script_path.read_bytes() == second.script_path.read_bytes()
    assert first.bundle_hash == second.bundle_hash
    assert code_hash(module.parents[1]) == before


def test_oversized_bundle_is_refused(toy_lab, tmp_path, caplog):
    noise = base64.b64encode(__import__("os").urandom(900_000)).decode()
    (toy_lab / "directions" / TOY / "src" / "big.txt").write_text(noise)
    with pytest.raises(BundleTooLargeError, match="big.txt"):
        build(toy_lab, tmp_path / "k")
    assert not (tmp_path / "k" / "kernel.py").exists()


def test_bundle_size_is_logged(toy_lab, tmp_path, caplog):
    with caplog.at_level("INFO", logger="labcore.bundle"):
        build(toy_lab, tmp_path / "k")
    assert "kernel script size" in caplog.text


def test_missing_entrypoint(toy_lab, tmp_path):
    (toy_lab / "directions" / TOY / "src" / "main.py").unlink()
    with pytest.raises(BundleError, match="entrypoint"):
        build(toy_lab, tmp_path / "k")


def test_the_real_direction_fits_the_limit(tmp_path):
    directory = REPO / "directions" / "isles24-ctp"
    info = build_kernel(
        tmp_path, direction_dir=directory, config_path=directory / "configs" / "exp_001_pilot.yaml",
        run_id="r1", exp="exp_001_pilot",
    )
    assert info.size_bytes < MAX_SCRIPT_BYTES // 2


# ---------------------------------------------------------------- entrypoint helpers


def test_only_missing_requirements_are_installed():
    options, missing = entrypoint_template.missing_requirements(
        "# comment\n--extra-index-url https://example.invalid/simple\nnumpy\npyyaml>=1  # inline\n"
        "numpy<0.1\nsurely-not-installed-package==1.0\nskipped; python_version < '2'\n"
    )
    assert options == ["--extra-index-url", "https://example.invalid/simple"]
    assert missing == ["numpy<0.1", "surely-not-installed-package==1.0"]


def test_unpack_refuses_paths_outside_the_bundle(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("../escape.txt")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError, match="unsafe"):
        entrypoint_template.unpack(base64.b64encode(buffer.getvalue()).decode(), tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()


# ---------------------------------------------------------------- metadata


def test_metadata():
    meta = build_metadata(
        username="Some-User", slug="isles24-ctp", exp="exp_001_pilot", code_file="kernel.py"
    )
    assert meta["id"] == "some-user/isles24-ctp-exp-001-pilot"
    assert meta["title"] == meta["id"].split("/")[1]  # the title resolves to the same slug
    assert meta["kernel_type"] == "script" and meta["language"] == "python"
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == []


@pytest.mark.parametrize("slug, exp", [("a", "b"), ("s" * 30, "e" * 30)])
def test_kernel_names_outside_kaggle_limits(slug, exp):
    with pytest.raises(ValueError):
        kernel_slug(slug, exp)


@pytest.mark.parametrize("username", ["", None, "a b", "x/y"])
def test_a_username_is_required(username):
    with pytest.raises(ValueError, match="KAGGLE_USERNAME"):
        build_metadata(username=username, slug="slug-a", exp="exp", code_file="k.py")


# ---------------------------------------------------------------- CLI wrapper


@pytest.mark.parametrize(
    "text, status",
    [
        ('u/k has status "KernelWorkerStatus.QUEUED"', KernelStatus.QUEUED),
        ('u/k has status "KernelWorkerStatus.RUNNING"', KernelStatus.RUNNING),
        ('u/k has status "KernelWorkerStatus.COMPLETE"', KernelStatus.COMPLETE),
        ('u/k has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', KernelStatus.CANCELLED),
        ('u/k has status "KernelWorkerStatus.CANCEL_REQUESTED"', KernelStatus.RUNNING),
        ('u/k has status "KernelWorkerStatus.NEW_SCRIPT"', KernelStatus.QUEUED),
        ('u/k has status "complete"', KernelStatus.COMPLETE),
        ('u/k has status "cancelAcknowledged"', KernelStatus.CANCELLED),
        ('u/k has status "something new"', KernelStatus.UNKNOWN),
        ("unexpected output", KernelStatus.UNKNOWN),
    ],
)
def test_parse_status(text, status):
    assert parse_status(text).status == status


def test_failure_message():
    result = parse_status('u/k has status "KernelWorkerStatus.ERROR"\nFailure message: "boom"')
    assert (result.status, result.failure_message) == (KernelStatus.ERROR, "boom")


def test_parse_push():
    assert parse_push("Kernel version 4 successfully pushed.  Please check progress at https://x") == 4
    for text in (
        "Kernel push error: Maximum weekly GPU quota reached",
        "The following are not valid dataset sources: ['a/b']\nKernel version 1 successfully pushed.",
        "something else entirely",
    ):
        with pytest.raises(KaggleError):
            parse_push(text)


def test_log_to_text():
    raw = json.dumps([{"stream_name": "stdout", "time": 1, "data": "a\n"}, {"data": "b\n"}])
    assert log_to_text(raw) == "a\nb\n"
    assert log_to_text("plain text") == "plain text"


class FakeCLI:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, command, **options):
        self.calls.append((command, options))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        code, text = reply
        return subprocess.CompletedProcess(command, code, stdout=text, stderr="")


def client(*replies):
    cli = FakeCLI(*replies)
    return KaggleClient(runner=cli, sleep=lambda seconds: None), cli


def kernel_dir(tmp_path):
    write_metadata(tmp_path, {"id": "user/toy-kernel", "code_file": "kernel.py"})
    return tmp_path


def test_push(tmp_path):
    kaggle, cli = client((0, "Kernel version 2 successfully pushed."))
    result = kaggle.push(kernel_dir(tmp_path), timeout_s=3600)
    assert (result.ref, result.version) == ("user/toy-kernel/2", 2)
    assert cli.calls[0][0][1:] == ["kernels", "push", "-p", str(tmp_path), "-t", "3600"]


def test_push_error_with_exit_code_zero_is_an_error(tmp_path):
    kaggle, cli = client((0, "Kernel push error: invalid accelerator"))
    with pytest.raises(KaggleError, match="invalid accelerator"):
        kaggle.push(kernel_dir(tmp_path))
    assert len(cli.calls) == 1


@pytest.mark.parametrize(
    "reply",
    [(1, "Read timed out."), (1, "503 Server Error: Service Unavailable"), subprocess.TimeoutExpired("k", 1)],
)
def test_push_is_not_repeated_when_it_may_have_arrived(tmp_path, reply):
    kaggle, cli = client(reply, (0, "Kernel version 1 successfully pushed."))
    with pytest.raises(KaggleError):
        kaggle.push(kernel_dir(tmp_path))
    assert len(cli.calls) == 1


def test_push_is_repeated_when_it_never_left(tmp_path):
    kaggle, cli = client(
        (1, "Failed to establish a new connection"), (0, "Kernel version 1 successfully pushed.")
    )
    assert kaggle.push(kernel_dir(tmp_path)).version == 1
    assert len(cli.calls) == 2


def test_status_retries_transient_errors_only():
    kaggle, cli = client((1, "503 Server Error"), (0, 'u/k has status "complete"'))
    assert kaggle.status("u/k").status == KernelStatus.COMPLETE
    assert len(cli.calls) == 2 and cli.calls[0][1]["timeout"] == 60

    kaggle, cli = client((1, "401 - Unauthorized"))
    with pytest.raises(KaggleError) as info:
        kaggle.status("u/k")
    assert not info.value.transient and len(cli.calls) == 1

    kaggle, cli = client(*[(1, "Connection reset by peer")] * 3)
    with pytest.raises(KaggleError) as info:
        kaggle.status("u/k")
    assert info.value.transient and len(cli.calls) == 3


def test_empty_secrets_are_not_passed_to_the_cli(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "")
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    kaggle, cli = client((0, 'u/k has status "running"'))
    kaggle.status("u/k")
    env = cli.calls[0][1]["env"]
    assert "KAGGLE_API_TOKEN" not in env and env["KAGGLE_USERNAME"] == "someone"


def test_missing_cli():
    kaggle, _ = client(FileNotFoundError("kaggle"))
    with pytest.raises(KaggleError, match="not found"):
        kaggle.status("u/k")


def test_output_reads_the_log_file(tmp_path):
    (tmp_path / "toy-kernel.log").write_text(json.dumps([{"data": "line 1\n"}]))
    kaggle, cli = client((0, "Output file downloaded"))
    result = kaggle.output("user/toy-kernel/3", tmp_path)
    assert result.log_text == "line 1\n"
    assert cli.calls[0][0][1:4] == ["kernels", "output", "user/toy-kernel/3"]


# ---------------------------------------------------------------- local backend


def push_local(root, tmp_path, monkeypatch=None):
    kernel = tmp_path / "kernel"
    build(root, kernel)
    write_metadata(
        kernel, build_metadata(username="local-user", slug=TOY, exp="exp_001", code_file="kernel.py")
    )
    backend = LocalBackend(tmp_path / "local")
    return backend, backend.push(kernel)


def test_local_backend_success(toy_lab, tmp_path):
    backend, pushed = push_local(toy_lab, tmp_path)
    assert pushed.ref == f"local-user/{TOY}-exp-001/1"
    assert backend.status(pushed.ref).status == KernelStatus.COMPLETE
    output = backend.output(pushed.ref, tmp_path / "out")
    status = json.loads((tmp_path / "out" / "out" / "status.json").read_text())
    env = json.loads((tmp_path / "out" / "out" / "env.json").read_text())
    assert status["ok"] and status["stage"] == "done" and status["run_id"] == "r1"
    assert [step["stage"] for step in status["history"]] == [
        "unpack", "requirements", "env", "run", "verify_outputs", "done",
    ]
    assert env["run_id"] == "r1" and env["config_hash"].startswith("sha256:") and "gpu_name" in env
    assert "stage=done" in output.log_text


def test_local_backend_failure_leaves_a_final_status(toy_lab, tmp_path):
    replace_pipeline(toy_lab, "raise ValueError('shapes differ')")
    backend, pushed = push_local(toy_lab, tmp_path)
    assert backend.status(pushed.ref).status == KernelStatus.ERROR
    backend.output(pushed.ref, tmp_path / "out")
    status = json.loads((tmp_path / "out" / "out" / "status.json").read_text())
    assert (status["ok"], status["stage"], status["error_type"]) == (False, "run", "ValueError")
    assert "shapes differ" in status["traceback_tail"] and status["finished_utc"]


def test_missing_outputs_fail_the_kernel(toy_lab, tmp_path):
    replace_pipeline(toy_lab, "pass")
    backend, pushed = push_local(toy_lab, tmp_path)
    assert backend.status(pushed.ref).status == KernelStatus.ERROR


def test_local_kernels_get_no_credentials(toy_lab, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder")
    monkeypatch.setenv("KAGGLE_KEY", "placeholder")
    replace_pipeline(
        toy_lab,
        "import os\nprint('LEAK' if {'DEEPSEEK_API_KEY', 'KAGGLE_KEY'} & set(os.environ) else 'CLEAN')",
    )
    backend, pushed = push_local(toy_lab, tmp_path)
    log = backend.output(pushed.ref, tmp_path / "out").log_text
    assert "CLEAN" in log and "LEAK" not in log
