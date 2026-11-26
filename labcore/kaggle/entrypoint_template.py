"""Kernel entrypoint. bundle.py fills in the two placeholders below and pushes this one file.

It must stay standalone: only the standard library may be used before the bundle is unpacked.
Everything the run produces goes to <working>/out/, and out/status.json is written at every
stage, so even a crashed run leaves a final status behind.
"""

import base64
import importlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_JSON = "{}"  # __LAB_MANIFEST__
PAYLOAD_B64 = ""  # __LAB_PAYLOAD__

TRACEBACK_TAIL_LINES = 60


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017


def log(message):
    print(f"[lab] {utc_now()} {message}", flush=True)


class Status:
    """out/status.json. `ok` is true only after every stage has finished successfully."""

    def __init__(self, path, run_id):
        self.path = path
        self.data = {
            "run_id": run_id,
            "stage": "start",
            "utc_time": utc_now(),
            "ok": False,
            "error_type": None,
            "error_message": None,
            "traceback_tail": None,
            "started_utc": utc_now(),
            "finished_utc": None,
            "history": [],
        }
        self.write()

    def write(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def enter(self, stage, ok=False):
        self.data.update(stage=stage, utc_time=utc_now(), ok=ok)
        self.data["history"].append({"stage": stage, "utc_time": self.data["utc_time"], "ok": ok})
        log(f"stage={stage}")
        self.write()

    def fail(self, error):
        lines = "".join(traceback.format_exception(type(error), error, error.__traceback__)).splitlines()
        self.data.update(
            ok=False,
            utc_time=utc_now(),
            error_type=type(error).__name__,
            error_message=str(error)[:4000],
            traceback_tail="\n".join(lines[-TRACEBACK_TAIL_LINES:]),
        )
        self.data["history"].append(
            {"stage": self.data["stage"], "utc_time": self.data["utc_time"], "ok": False}
        )
        self.write()

    def finish(self):
        self.data["finished_utc"] = utc_now()
        self.write()


def unpack(payload_b64, destination):
    raw = base64.b64decode("".join(payload_b64.split()))
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if not (member.isfile() or member.isdir()) or not target.is_relative_to(root):
                raise RuntimeError(f"unsafe entry in bundle: {member.name}")
        if hasattr(tarfile, "data_filter"):
            archive.extractall(root, filter="data")
        else:
            archive.extractall(root)


def missing_requirements(text):
    """Return (pip option lines, requirement lines that are not satisfied yet)."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None
    options, missing = [], []
    for raw in text.splitlines():
        line = re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):
            options.extend(line.split())
            continue
        name, specifier, marker = re.split(r"[\s<>=!~;\[@]", line, maxsplit=1)[0], None, None
        if Requirement is not None:
            parsed = Requirement(line)
            name, specifier, marker = parsed.name, parsed.specifier, parsed.marker
        if marker is not None and not marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(line)
            continue
        if specifier is not None and not specifier.contains(version, prereleases=True):
            missing.append(line)
    return options, missing


def install_requirements(path):
    if not path.is_file():
        log("no requirements.txt")
        return
    options, missing = missing_requirements(path.read_text(encoding="utf-8"))
    if not missing:
        log("all requirements already installed")
        return
    if os.environ.get("LAB_SKIP_PIP"):
        raise RuntimeError(f"missing requirements (pip is disabled for local runs): {', '.join(missing)}")
    # UNVERIFIED: pip can reach the package index from the kernel (needs internet enabled).
    log(f"installing: {', '.join(missing)}")
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *options, *missing]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    print(done.stdout, flush=True)
    print(done.stderr, file=sys.stderr, flush=True)
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-15:])
        raise RuntimeError(f"pip install failed (exit {done.returncode}):\n{tail}")
    importlib.invalidate_caches()


def environment(manifest):
    info = {
        key: manifest.get(key)
        for key in ("run_id", "slug", "exp", "commit_sha", "config_hash", "bundle_hash", "labcore_version")
    }
    info.update(
        python_version=platform.python_version(), platform=platform.platform(), gpu_name=None,
        torch_version=None,
    )
    try:
        info["torch_version"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return info
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as error:  # the environment record must never kill the run
        info["torch_error"] = f"{type(error).__name__}: {error}"
    return info


def main():
    manifest = json.loads(MANIFEST_JSON)
    # UNVERIFIED: files under /kaggle/working are what `kernels output` returns.
    working = Path(os.environ.get("LAB_KAGGLE_WORKING", "/kaggle/working"))
    out = working / "out"
    out.mkdir(parents=True, exist_ok=True)
    status = Status(out / "status.json", manifest.get("run_id"))
    try:
        log(f"run_id={manifest.get('run_id')} commit={manifest.get('commit_sha')}")
        status.enter("unpack")
        bundle = Path(tempfile.mkdtemp(prefix="lab_bundle_"))
        unpack(PAYLOAD_B64, bundle)
        direction = bundle / "directions" / manifest["slug"]

        status.enter("requirements")
        install_requirements(direction / "requirements.txt")

        status.enter("env")
        sys.path.insert(0, str(bundle))
        sys.path.insert(0, str(direction))
        info = environment(manifest)
        (out / "env.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        log(f"gpu={info['gpu_name']} torch={info['torch_version']}")

        status.enter("run")
        module = importlib.import_module(manifest["module"])
        entry = getattr(module, manifest["function"])
        entry(str(bundle / manifest["config_path"]), str(out))

        status.enter("verify_outputs")
        absent = [name for name in manifest["outputs"] if not (out / name).is_file()]
        if absent:
            raise RuntimeError(f"required outputs were not written: {', '.join(absent)}")

        status.enter("done", ok=True)
    except BaseException as error:
        status.fail(error)
        raise  # a non-zero exit marks the kernel as failed
    finally:
        status.finish()


if __name__ == "__main__":
    main()
