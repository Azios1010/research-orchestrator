"""Build the standard scorecard from a run's output files, following `scorecard_map`."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from labcore.io import parse_iso, read_json
from labcore.schema import schema_errors

COMMON_FIELDS = ("n_subjects_requested", "n_subjects_usable", "pipeline_ok", "gpu_hours", "runtime_min")


class ScorecardError(RuntimeError):
    """A mapped value could not be read from the outputs."""


def _resolve_file(result_dir: Path, name: str) -> Path:
    path = (result_dir / name).resolve()
    if not path.is_relative_to(result_dir.resolve()):
        raise ScorecardError(f"scorecard source {name!r} points outside the result folder")
    if not path.is_file():
        raise ScorecardError(f"scorecard source file is missing: {name}")
    return path


def read_source(result_dir: Path, source: Mapping[str, Any]) -> Any:
    """Read one value. JSON sources use a dotted `path`; CSV sources use `where`/`column`/`agg`."""
    path = _resolve_file(Path(result_dir), source["file"])
    if "path" in source:
        value: Any = read_json(path)
        for part in source["path"].split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                raise ScorecardError(f"{source['file']}: no value at path {source['path']!r}")
        return value
    frame = pd.read_csv(path)
    for column, wanted in source.get("where", {}).items():
        if column not in frame:
            raise ScorecardError(f"{source['file']}: no column {column!r}")
        frame = frame[frame[column].astype(str) == str(wanted)]
    column = source["column"]
    if column not in frame:
        raise ScorecardError(f"{source['file']}: no column {column!r}")
    if "agg" in source:
        if source["agg"] == "count":
            return int(frame[column].count())
        if frame.empty:
            raise ScorecardError(f"{source['file']}: no rows match {source.get('where')}")
        return float(getattr(frame[column], source["agg"])())
    if len(frame) != 1:
        raise ScorecardError(
            f"{source['file']}: expected exactly one row for {source.get('where')}, found {len(frame)}"
        )
    value = frame[column].iloc[0]
    return value.item() if hasattr(value, "item") else value


def runtime_from_status(
    status: Mapping[str, Any] | None, env: Mapping[str, Any] | None
) -> tuple[float, float]:
    """(runtime_min, gpu_hours) from the kernel's own timestamps. CPU-only runs use no GPU hours."""
    if not status or not status.get("started_utc"):
        return 0.0, 0.0
    end = status.get("finished_utc") or status.get("utc_time")
    seconds = max(0.0, (parse_iso(end) - parse_iso(status["started_utc"])).total_seconds())
    minutes = seconds / 60.0
    gpu_hours = minutes / 60.0 if env and env.get("gpu_name") else 0.0
    return round(minutes, 3), round(gpu_hours, 4)


def build_scorecard(
    result_dir: Path,
    scorecard_map: Mapping[str, Any],
    run_meta: Mapping[str, str],
    *,
    pipeline_ok: bool,
) -> dict:
    result_dir = Path(result_dir)
    runtime_min, gpu_hours = runtime_from_status(
        read_json(result_dir / "status.json"), read_json(result_dir / "env.json")
    )
    identity = ("slug", "exp", "run_id", "commit_sha", "config_hash")
    scorecard: dict[str, Any] = {key: run_meta[key] for key in identity}
    for key in ("n_subjects_requested", "n_subjects_usable"):
        value = read_source(result_dir, scorecard_map[key])
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        scorecard[key] = value
    scorecard["pipeline_ok"] = bool(pipeline_ok)
    scorecard["gpu_hours"] = gpu_hours
    scorecard["runtime_min"] = runtime_min
    scorecard["metrics"] = {
        name: read_source(result_dir, source) for name, source in scorecard_map["metrics"].items()
    }
    return scorecard


def scorecard_problems(scorecard: Mapping[str, Any]) -> list[str]:
    """Missing, NaN or infinite values, then schema violations."""
    problems = []
    for name, value in flatten(scorecard).items():
        if value is None:
            problems.append(f"scorecard field {name} is missing")
        elif isinstance(value, float) and not math.isfinite(value):
            problems.append(f"scorecard field {name} is NaN or infinite")
    if problems:
        return problems
    return [f"scorecard schema: {error}" for error in schema_errors(dict(scorecard), "scorecard")]


def flatten(scorecard: Mapping[str, Any]) -> dict[str, Any]:
    """Common fields plus metrics: the namespace that gate expressions can name."""
    values = {name: scorecard.get(name) for name in COMMON_FIELDS}
    values.update(scorecard.get("metrics", {}))
    return values
