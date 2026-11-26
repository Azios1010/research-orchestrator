"""state/runs.json, state/queue.json and state/gpu_budget.json."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from labcore.io import iso, parse_iso, read_json, write_json
from labcore.schema import schema_errors

ACTIVE_STATUSES = ("QUEUED", "RUNNING", "UNKNOWN")
HARD_TIMEOUT_HOURS = 11.5
TIMEOUT_FACTOR = 2.0
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


class StateError(RuntimeError):
    pass


def timeout_hours(est_gpu_hours: float) -> float:
    """A run is overdue after twice its estimate, and never allowed more than 11.5 hours."""
    return min(TIMEOUT_FACTOR * est_gpu_hours, HARD_TIMEOUT_HOURS)


def default_budget() -> dict[str, Any]:
    # UNVERIFIED: Kaggle grants about 30 GPU hours a week; run `kaggle quota` to see the real
    # allowance and refresh time, then adjust weekly_cap_hours / reset_weekday.
    return {
        "weekly_cap_hours": 25,
        "reset_weekday": "saturday",
        "week_start": None,
        "used_hours": 0,
        "entries": [],
    }


class StateStore:
    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)

    def _load(self, name: str, default: dict, definition: str | None) -> dict:
        path = self.dir / name
        document = read_json(path, default)
        errors = schema_errors(document, "runs", definition)
        if errors:
            raise StateError(f"{path} is invalid:\n  " + "\n  ".join(errors))
        return document

    def _save(self, name: str, document: dict, definition: str | None) -> None:
        errors = schema_errors(document, "runs", definition)
        if errors:
            raise StateError(f"refusing to write an invalid {name}:\n  " + "\n  ".join(errors))
        write_json(self.dir / name, document)

    def runs(self) -> list[dict]:
        return self._load("runs.json", {"runs": []}, None)["runs"]

    def save_runs(self, runs: list[dict]) -> None:
        self._save("runs.json", {"runs": runs}, None)

    def put_run(self, record: dict) -> None:
        runs = self.runs()
        for index, existing in enumerate(runs):
            if existing["run_id"] == record["run_id"]:
                runs[index] = record
                break
        else:
            runs.append(record)
        self.save_runs(runs)

    def active_runs(self) -> list[dict]:
        return [r for r in self.runs() if r["status"] in ACTIVE_STATUSES]

    def queue(self) -> list[dict]:
        return self._load("queue.json", {"queue": []}, "queue")["queue"]

    def save_queue(self, queue: list[dict]) -> None:
        self._save("queue.json", {"queue": queue}, "queue")

    def budget(self) -> dict:
        return self._load("gpu_budget.json", default_budget(), "budget")

    def save_budget(self, budget: dict) -> None:
        budget["used_hours"] = used_hours(budget)
        self._save("gpu_budget.json", budget, "budget")


def week_start(now: datetime, reset_weekday: str) -> str:
    back = (now.weekday() - WEEKDAYS.index(reset_weekday)) % 7
    return (now - timedelta(days=back)).date().isoformat()


def roll_week(budget: dict, now: datetime) -> dict:
    """At a new quota week, forget finished runs but keep the reservations of unfinished ones."""
    current = week_start(now, budget["reset_weekday"])
    if budget["week_start"] != current:
        budget["entries"] = [e for e in budget["entries"] if not e["settled"]]
        budget["week_start"] = current
    return budget


def used_hours(budget: dict) -> float:
    return round(sum(entry["hours"] for entry in budget["entries"]), 4)


def reserve(budget: dict, run_id: str, slug: str, hours: float, now: datetime) -> None:
    roll_week(budget, now)
    budget["entries"].append(
        {"run_id": run_id, "slug": slug, "hours": hours, "settled": False, "at": iso(now)}
    )


def settle(budget: dict, run_id: str, slug: str, hours: float, finished_at: str, now: datetime) -> None:
    """Replace a reservation with the hours actually used."""
    roll_week(budget, now)
    budget["entries"] = [e for e in budget["entries"] if e["run_id"] != run_id]
    if parse_iso(finished_at).date().isoformat() >= budget["week_start"]:
        budget["entries"].append(
            {"run_id": run_id, "slug": slug, "hours": round(hours, 4), "settled": True, "at": finished_at}
        )


def direction_hours(runs: list[dict], slug: str) -> float:
    """GPU hours this direction has used or reserved over all its runs."""
    total = 0.0
    for run in runs:
        if run["slug"] == slug:
            actual = run.get("actual_gpu_hours")
            total += run["est_gpu_hours"] if actual is None else actual
    return round(total, 4)
