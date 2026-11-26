"""Compute a direction's verdict from its locked gates and write the top of VERDICT.md.

    python tools/verdict.py <slug>             # record the verdict of the current L1 round
    python tools/verdict.py <slug> --dry-run   # preview on .dryrun/ results; changes nothing tracked
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labcore import registry
from labcore.config import ConfigError, load_direction
from labcore.io import iso, read_json, utcnow
from labcore.lab import DRYRUN_DIRNAME, Lab
from labcore.runs.state import StateError, StateStore
from labcore.stats.scorecard import scorecard_problems
from labcore.verdict.gates import GateError
from labcore.verdict.prereg import LOCKED_TEXT, hash_for_entry
from labcore.verdict.verdict import (
    VerdictError,
    VerdictResult,
    compute_verdict,
    render_block,
    update_verdict_md,
)

DRY_RUN_BANNER = "DRY RUN on fake data with unlocked gates. This is not a result."


def _scorecard(lab: Lab, run: dict) -> dict:
    path = lab.results_dir / run["slug"] / run["exp"] / run["run_id"] / "scorecard.json"
    scorecard = read_json(path)
    if scorecard is None:
        raise VerdictError(f"no scorecard at {path}")
    problems = scorecard_problems(scorecard)
    if problems or scorecard["run_id"] != run["run_id"]:
        raise VerdictError(f"{path} is invalid: {problems or 'run_id mismatch'}")
    return scorecard


def _pick(runs: list[dict], run_id: str | None, slug: str, why_empty: str) -> dict:
    if not runs:
        raise VerdictError(f"{slug}: no usable run. {why_empty}")
    if run_id is None:
        return runs[-1]
    for run in runs:
        if run["run_id"] == run_id:
            return run
    raise VerdictError(f"{run_id} is not among the usable runs: {[r['run_id'] for r in runs]}")


def record_verdict(
    lab: Lab, slug: str, run_id: str | None = None, allow_local: bool = False
) -> VerdictResult:
    """The real thing: requires an intact lock, writes VERDICT.md and sets the registry status."""
    item = registry.entry(registry.load(lab.registry_path), slug)
    directory = lab.direction_dir(slug)
    if item["status"] not in ("L1_RUNNING", "VERDICT"):
        raise VerdictError(f"{slug} is {item['status']}: verdicts exist only for directions in L1")
    lock = item.get("prereg_lock") or {}
    try:
        current = hash_for_entry(directory, item)
    except (OSError, ValueError) as exc:
        raise VerdictError(f"{slug}: the lock cannot be checked: {exc}. No verdict.") from exc
    if lock.get("hash") != current:
        raise VerdictError(f"{slug}: {LOCKED_TEXT} differs from the lock. No verdict.")
    direction = load_direction(directory)
    usable = sorted(
        (
            run for run in StateStore(lab.state_dir).runs()
            if run["slug"] == slug and run["status"] == "COMPLETE"
            and run.get("prereg_hash") == lock["hash"] and run.get("l1_round") == lock["round"]
            and (run.get("backend") == "kaggle" or allow_local)
        ),
        key=lambda run: run["submitted_at"],
    )
    chosen = _pick(usable, run_id, slug, "A run must be COMPLETE and submitted under the current lock.")
    scorecard = _scorecard(lab, chosen)
    result = compute_verdict(direction["gates"], scorecard)
    others = [
        (run["run_id"], compute_verdict(direction["gates"], _scorecard(lab, run)).verdict)
        for run in usable if run["run_id"] != chosen["run_id"]
    ]
    block = render_block(
        result, scorecard=scorecard, prereg_hash=lock["hash"], l1_round=lock["round"],
        computed_at=iso(utcnow()), other_runs=others,
    )
    update_verdict_md(directory / "VERDICT.md", block, direction["title"])
    registry.set_status(lab.root, slug, "VERDICT", verdict=result.verdict)
    return result


def preview_verdict(lab: Lab, slug: str, run_id: str | None = None) -> tuple[VerdictResult, Path]:
    """Apply the current gates to a dry-run result. Registry and VERDICT.md stay untouched."""
    direction = load_direction(lab.direction_dir(slug))
    runs = sorted(
        (r for r in StateStore(lab.state_dir).runs() if r["slug"] == slug and r["status"] == "COMPLETE"),
        key=lambda run: run["submitted_at"],
    )
    chosen = _pick(runs, run_id, slug, "Run the dry-run dispatch and watcher first.")
    scorecard = _scorecard(lab, chosen)
    result = compute_verdict(direction["gates"], scorecard)
    block = render_block(
        result, scorecard=scorecard, prereg_hash=chosen.get("prereg_hash"), l1_round=chosen.get("l1_round"),
        computed_at=iso(utcnow()), banner=DRY_RUN_BANNER,
    )
    path = lab.root / DRYRUN_DIRNAME / f"VERDICT.{slug}.md"
    update_verdict_md(path, block, direction["title"])
    return result, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("slug")
    parser.add_argument("--run-id", help="use this run instead of the latest usable one")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local", action="store_true", help="accept local-backend runs (testing only)")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--state-dir")
    parser.add_argument("--results-dir")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    base = Lab.dry_run(root) if args.dry_run else Lab.at(root)
    lab = Lab.at(root, args.state_dir or base.state_dir, args.results_dir or base.results_dir)
    try:
        if args.dry_run:
            result, path = preview_verdict(lab, args.slug, args.run_id)
            print(f"dry run: {result.verdict} (written to {path})")
        else:
            result = record_verdict(lab, args.slug, args.run_id, args.allow_local)
            print(f"{args.slug}: {result.verdict}")
            print("Write the interpretation below the marker in VERDICT.md, then open a pull request.")
    except (VerdictError, GateError, ConfigError, StateError, registry.RegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
