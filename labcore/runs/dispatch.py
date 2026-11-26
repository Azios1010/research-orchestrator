"""Submit one experiment: validate, check the lock and the budget, bundle, push, record.

    python -m labcore.runs.dispatch <slug> <exp>              # real Kaggle push (needs the lock)
    python -m labcore.runs.dispatch <slug> <exp> --dry-run    # fake data, local CPU, .dryrun/
    python -m labcore.runs.dispatch --drain-queue
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from labcore import registry
from labcore.config import ConfigError, experiment_path, load_direction, load_experiment
from labcore.gitutil import head_sha
from labcore.io import iso, utcnow, write_text_atomic
from labcore.kaggle.bundle import BundleError, build_kernel, code_hash
from labcore.kaggle.client import Backend, KaggleClient, KaggleError
from labcore.kaggle.local_backend import LocalBackend
from labcore.kaggle.metadata import build_metadata, write_metadata
from labcore.lab import DRYRUN_DIRNAME, Lab, find_root
from labcore.modules import load_module
from labcore.runs import state as st
from labcore.verdict.prereg import LOCKED_TEXT, prereg_hash

LOCAL_USERNAME = "local-user"


class DispatchError(RuntimeError):
    pass


class NotLockedError(DispatchError):
    pass


class BudgetError(DispatchError):
    pass


@dataclass
class Outcome:
    action: str  # "submitted" | "queued" | "skipped"
    message: str
    record: dict | None = None


def lock_state(source_root: Path, slug: str) -> tuple[str, int]:
    """(prereg hash, L1 round) if the direction is locked and unchanged, else NotLockedError."""
    item = registry.entry(registry.load(source_root / "registry.yaml"), slug)
    direction_dir = source_root / "directions" / slug
    if item["status"] != "L1_RUNNING":
        raise NotLockedError(
            f"{slug} is {item['status']}, not L1_RUNNING. Run L0, get the gates approved, then "
            f"`python tools/lock_prereg.py {slug}`."
        )
    if load_direction(direction_dir)["gates"]["approval"] != "APPROVED":
        raise NotLockedError(f"{slug}: gates are not APPROVED")
    lock = item.get("prereg_lock") or {}
    try:
        current = prereg_hash(direction_dir, scheme=lock.get("hash_scheme", 1))
    except (OSError, ValueError) as exc:
        raise NotLockedError(f"{slug}: the lock cannot be checked: {exc}") from exc
    if lock.get("hash") != current:
        raise NotLockedError(f"{slug}: {LOCKED_TEXT} changed after the lock")
    return current, lock["round"]


def new_run_id(slug: str, exp: str, now: datetime) -> str:
    return f"{slug}-{exp}-{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}".replace("_", "-").lower()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def dispatch(
    lab: Lab,
    slug: str,
    exp: str,
    backend: Backend,
    *,
    username: str | None = None,
    require_lock: bool | None = None,
    enqueue: bool = False,
    force: bool = False,
    config_path: Path | None = None,
    source_root: Path | None = None,
    commit_sha: str | None = None,
    retries: int = 0,
    retry_of: str | None = None,
    issue_number: int | None = None,
    expected_bundle_hash: str | None = None,
    now: datetime | None = None,
) -> Outcome:
    """`source_root` is the checkout the code is taken from (a snapshot, for retries)."""
    now = now or utcnow()
    source_root = Path(source_root or lab.root)
    if require_lock is None:
        require_lock = backend.name == "kaggle"
    username = username or (LOCAL_USERNAME if backend.name == "local" else os.environ.get("KAGGLE_USERNAME"))

    registry.entry(registry.load(source_root / "registry.yaml"), slug)
    direction_dir = source_root / "directions" / slug
    direction = load_direction(direction_dir)
    if direction["slug"] != slug:
        raise DispatchError(f"direction.yaml says slug {direction['slug']!r}, folder is {slug!r}")
    config_path = Path(config_path) if config_path else experiment_path(direction_dir, exp)
    estimate = float(load_experiment(config_path)["compute"]["est_gpu_hours"])

    try:
        locked_hash, l1_round = lock_state(source_root, slug)
    except NotLockedError:
        if require_lock:
            raise
        locked_hash, l1_round = None, None

    metadata = build_metadata(
        username=username or "",
        slug=slug,
        exp=exp,
        code_file="kernel.py",
        dataset_sources=direction["data"].get("dataset_sources"),
    )
    store = st.StateStore(lab.state_dir)
    runs = store.runs()
    fingerprint = code_hash(direction_dir)

    with tempfile.TemporaryDirectory(prefix="lab_kernel_") as temporary:
        kernel_dir = Path(temporary)
        run_id = new_run_id(slug, exp, now)
        sha = commit_sha or head_sha(source_root, ("labcore", f"directions/{slug}", "registry.yaml"))
        bundle = build_kernel(
            kernel_dir,
            direction_dir=direction_dir,
            config_path=config_path,
            run_id=run_id,
            exp=exp,
            commit_sha=sha,
            labcore_dir=source_root / "labcore",
        )
        if expected_bundle_hash and bundle.bundle_hash != expected_bundle_hash:
            raise DispatchError("the code differs from the original submission; not repeating it")

        key = (slug, exp, bundle.config_hash, fingerprint, l1_round)

        def identical(run: dict) -> bool:
            return (
                run["slug"], run["exp"], run["config_hash"], run.get("code_hash"), run.get("l1_round")
            ) == key

        # One kernel holds one run at a time: a second push would replace the running version.
        active = [r for r in runs if r["kernel_id"] == metadata["id"] and r["status"] in st.ACTIVE_STATUSES]
        if active and not (force and backend.name == "local"):
            if identical(active[0]) and not force:
                return Outcome("skipped", f"identical run already active: {active[0]['run_id']}")
            raise DispatchError(f"run {active[0]['run_id']} is still active on kernel {metadata['id']}")
        done = [r for r in runs if r["status"] == "COMPLETE" and identical(r)]
        if done and not force:
            previous = done[-1]["run_id"]
            return Outcome("skipped", f"identical run already completed: {previous} (--force to repeat)")

        cap = float(direction["compute"]["l1_gpu_hours_cap"])
        spent = st.direction_hours(runs, slug)
        if spent + estimate > cap + 1e-9:
            raise BudgetError(
                f"{slug}: {spent:g} h used + {estimate:g} h estimated exceeds l1_gpu_hours_cap {cap:g} h"
            )
        budget = st.roll_week(store.budget(), now)
        weekly, used = float(budget["weekly_cap_hours"]), st.used_hours(budget)
        if used + estimate > weekly + 1e-9:
            reason = f"weekly GPU cap: {used:g} h used + {estimate:g} h estimated > {weekly:g} h"
            if enqueue and estimate <= weekly:
                queue = [q for q in store.queue() if (q["slug"], q["exp"]) != (slug, exp)]
                queue.append({"slug": slug, "exp": exp, "enqueued_at": iso(now), "reason": reason})
                store.save_queue(queue)
                return Outcome("queued", f"queued ({reason})")
            raise BudgetError(reason)

        write_metadata(kernel_dir, metadata)
        pushed = backend.push(kernel_dir, timeout_s=int(st.timeout_hours(estimate) * 3600))

    record = {
        "run_id": run_id,
        "slug": slug,
        "exp": exp,
        "kernel_id": metadata["id"],
        "kernel_version": pushed.version,
        "backend": backend.name,
        "commit_sha": sha,
        "config_hash": bundle.config_hash,
        "code_hash": fingerprint,
        "bundle_hash": bundle.bundle_hash,
        "prereg_hash": locked_hash,
        "l1_round": l1_round,
        "status": "QUEUED",
        "submitted_at": iso(now),
        "last_checked_at": None,
        "finished_at": None,
        "est_gpu_hours": estimate,
        "actual_gpu_hours": None,
        "retries": retries,
        "retry_of": retry_of,
        "result_path": _relative(lab.results_dir / slug / exp / run_id, lab.root),
        "issue_number": issue_number,
        "failure_class": None,
    }
    store.put_run(record)
    st.reserve(budget, run_id, slug, estimate, now)
    store.save_budget(budget)
    store.save_queue([q for q in store.queue() if (q["slug"], q["exp"]) != (slug, exp)])
    return Outcome("submitted", f"submitted {run_id} to {metadata['id']} ({bundle.size_bytes} bytes)", record)


def drain_queue(lab: Lab, backend: Backend, **options) -> list[str]:
    """Submit queued experiments in order while the budget allows. Entries that fail stay queued."""
    store = st.StateStore(lab.state_dir)
    report = []
    for item in list(store.queue()):
        label = f"{item['slug']}/{item['exp']}"
        try:
            outcome = dispatch(lab, item["slug"], item["exp"], backend, **options)
        except (DispatchError, ConfigError, KaggleError, BundleError, registry.RegistryError) as exc:
            report.append(f"{label}: still queued: {exc}")
            continue
        report.append(f"{label}: {outcome.message}")
        store.save_queue([q for q in store.queue() if (q["slug"], q["exp"]) != (item["slug"], item["exp"])])
    return report


def dry_run_config(lab: Lab, slug: str, exp: str) -> Path:
    """The experiment config with the direction's fake data and tiny settings swapped in."""
    direction_dir = lab.direction_dir(slug)
    base = load_experiment(experiment_path(direction_dir, exp))
    work = lab.root / DRYRUN_DIRNAME / "work" / slug / exp
    work.mkdir(parents=True, exist_ok=True)
    fake = load_module(direction_dir / "tests" / "fake_data.py")
    config = fake.make_dryrun_config(work, base)
    path = work / f"{exp}.yaml"
    write_text_atomic(path, yaml.safe_dump(config, sort_keys=False))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dispatch", description=__doc__.split("\n")[0])
    parser.add_argument("slug", nargs="?")
    parser.add_argument("exp", nargs="?")
    parser.add_argument("--dry-run", action="store_true", help="local CPU backend, fake data, .dryrun/")
    parser.add_argument("--real-config", action="store_true", help="dry run with the unmodified config")
    parser.add_argument("--enqueue", action="store_true", help="queue the run if the weekly budget is short")
    parser.add_argument("--force", action="store_true", help="repeat an identical completed run")
    parser.add_argument("--skip-unlocked", action="store_true", help="exit 0 if the direction is not locked")
    parser.add_argument("--drain-queue", action="store_true", help="submit queued runs")
    parser.add_argument("--root", help="lab root (default: found from the current folder)")
    parser.add_argument("--state-dir")
    parser.add_argument("--results-dir")
    parser.add_argument(
        "--source-root",
        help="checkout the kernel code and its lock are read from; state stays under --root. "
        "Nothing in it is executed here.",
    )
    args = parser.parse_args(argv)
    if not args.drain_queue and not (args.slug and args.exp):
        parser.error("slug and exp are required (or use --drain-queue)")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    root = Path(args.root).resolve() if args.root else find_root()
    if args.dry_run:
        default = Lab.dry_run(root)
        lab = Lab.at(root, args.state_dir or default.state_dir, args.results_dir or default.results_dir)
        backend: Backend = LocalBackend(root / DRYRUN_DIRNAME / "kernels")
    else:
        lab = Lab.at(root, args.state_dir, args.results_dir)
        backend = KaggleClient()

    source = {"source_root": Path(args.source_root).resolve()} if args.source_root else {}
    if source and args.dry_run:
        parser.error("--source-root cannot be combined with --dry-run (fake data code would be executed)")

    try:
        if args.drain_queue:
            lines = drain_queue(lab, backend, require_lock=not args.dry_run, **source)
            for line in lines or ["queue is empty"]:
                print(line)
            return 0
        options: dict = {"enqueue": args.enqueue, "force": args.force, **source}
        if args.dry_run:
            options.update(require_lock=False, force=True)
            if not args.real_config:
                options["config_path"] = dry_run_config(lab, args.slug, args.exp)
        outcome = dispatch(lab, args.slug, args.exp, backend, **options)
    except NotLockedError as exc:
        print(f"not dispatched: {exc}", file=sys.stderr)
        return 0 if args.skip_unlocked else 1
    except (DispatchError, ConfigError, KaggleError, BundleError, registry.RegistryError, ValueError) as exc:
        print(f"dispatch failed: {exc}", file=sys.stderr)
        return 1
    print(outcome.message)
    if args.dry_run:
        print("dry run: state and results are under .dryrun/.")
        print("next: python -m labcore.watcher.watcher --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
