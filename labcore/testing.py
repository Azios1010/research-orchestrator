"""Helpers for a direction's L0 smoke test. They work in the lab and in a graduated repo.

L0 goes through the real kernel path: bundle, then run the bundled script on CPU.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path

import yaml

from labcore.config import load_direction, load_experiment
from labcore.io import read_json
from labcore.kaggle.bundle import BundleInfo, build_kernel
from labcore.kaggle.client import KernelStatus
from labcore.kaggle.local_backend import LocalBackend
from labcore.kaggle.metadata import build_metadata, write_metadata
from labcore.modules import load_module, load_package
from labcore.stats.scorecard import build_scorecard, scorecard_problems
from labcore.watcher.checks import direction_checks, generic_checks

__all__ = [
    "L0Run", "experiment_params", "l0_problems", "l0_scorecard", "load_module", "load_package", "run_l0",
]


@dataclass
class L0Run:
    direction_dir: Path
    direction: dict
    out_dir: Path
    status: dict
    env: dict | None
    kernel_status: KernelStatus
    log: str
    bundle: BundleInfo


def experiment_params(direction_dir: Path, exp: str) -> dict:
    """The `params` of a real experiment config, ready to be overridden by a test."""
    return dict(load_experiment(Path(direction_dir) / "configs" / f"{exp}.yaml")["params"])


def run_l0(
    direction_dir: Path, params: dict, work_dir: Path, *, exp: str = "exp-l0", timeout_s: int = 900
) -> L0Run:
    direction_dir, work_dir = Path(direction_dir), Path(work_dir)
    direction = load_direction(direction_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / f"{exp}.yaml"
    config_path.write_text(
        yaml.safe_dump({"compute": {"est_gpu_hours": 0.1}, "params": params}, sort_keys=False),
        encoding="utf-8",
    )
    kernel_dir = work_dir / f"kernel-{secrets.token_hex(3)}"
    bundle = build_kernel(
        kernel_dir, direction_dir=direction_dir, config_path=config_path,
        run_id=f"l0-{secrets.token_hex(4)}", exp=exp, commit_sha="l0",
    )
    write_metadata(
        kernel_dir,
        build_metadata(username="local-user", slug=direction["slug"], exp=exp, code_file="kernel.py"),
    )
    backend = LocalBackend(work_dir / "kernels", timeout_s=timeout_s)
    pushed = backend.push(kernel_dir)
    result = backend.status(pushed.ref)
    output = backend.output(pushed.ref, work_dir / f"output-{pushed.version}")
    out_dir = output.dest / "out"
    return L0Run(
        direction_dir, direction, out_dir, read_json(out_dir / "status.json", {}),
        read_json(out_dir / "env.json"), result.status, output.log_text, bundle,
    )


def l0_scorecard(run: L0Run) -> dict:
    meta = {
        "slug": run.direction["slug"], "exp": "exp-l0", "run_id": run.status["run_id"],
        "commit_sha": "l0", "config_hash": run.bundle.config_hash,
    }
    scorecard = build_scorecard(
        run.out_dir, run.direction["scorecard_map"], meta, pipeline_ok=run.status.get("ok") is True
    )
    problems = scorecard_problems(scorecard)
    if problems:
        raise AssertionError("invalid scorecard: " + "; ".join(problems))
    return scorecard


def l0_problems(run: L0Run) -> list[str]:
    """Generic and direction-specific sanity checks, exactly as the watcher applies them."""
    problems = generic_checks(run.out_dir, run.direction["outputs"], run.direction.get("results_columns"))
    return problems + direction_checks(run.direction_dir / "checks.py", run.out_dir)
