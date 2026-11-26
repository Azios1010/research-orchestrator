"""Compute a verdict from locked gates and write it into VERDICT.md.

The verdict is a pure function of (gates, scorecard). No model is involved. Text below the
interpretation marker in VERDICT.md belongs to people (or Claude) and is never overwritten.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labcore.io import write_text_atomic
from labcore.stats.scorecard import flatten
from labcore.verdict import gates as gate_parser

BEGIN = "<!-- VERDICT:AUTO:BEGIN (written by tools/verdict.py; do not edit by hand) -->"
END = "<!-- VERDICT:AUTO:END -->"
INTERPRETATION = "<!-- VERDICT:INTERPRETATION (human or Claude; tools never overwrite below this line) -->"

DEFAULT_INTERPRETATION = """
## Interpretation

_Not written yet. Label every claim FACT / INTERPRETATION / HYPOTHESIS / UNKNOWN._

## Threats to validity

_Not written yet._
"""


class VerdictError(RuntimeError):
    """A verdict cannot be computed from this input."""


@dataclass
class VerdictResult:
    verdict: str  # "GO" | "PARK" | "REVISE"
    gates: dict[str, dict[str, Any]]
    values: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "gates": self.gates, "values": self.values, "notes": self.notes}


def compute_verdict(gates: Mapping[str, str], scorecard: Mapping[str, Any]) -> VerdictResult:
    """GO if the GO gate holds, PARK if the PARK gate holds, otherwise REVISE."""
    if scorecard.get("pipeline_ok") is not True:
        raise VerdictError("pipeline_ok is not true: an invalid run never produces a verdict")
    values = flatten(scorecard)
    evaluated: dict[str, dict[str, Any]] = {}
    used: dict[str, Any] = {}
    for name in ("GO", "PARK"):
        node = gate_parser.parse(gates[name])
        evaluated[name] = {"expression": gates[name], "result": gate_parser.evaluate(node, values)}
        used.update({key: values[key] for key in sorted(gate_parser.fields(node))})
    go, park = evaluated["GO"]["result"], evaluated["PARK"]["result"]
    notes: list[str] = []
    if go and park:
        verdict = "REVISE"
        notes.append("GO and PARK are both true: the gates conflict, so the result is REVISE.")
    elif go:
        verdict = "GO"
    elif park:
        verdict = "PARK"
    else:
        verdict = "REVISE"
    return VerdictResult(verdict, evaluated, used, notes)


def _format(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def render_block(
    result: VerdictResult,
    *,
    scorecard: Mapping[str, Any],
    prereg_hash: str | None,
    l1_round: int | None,
    computed_at: str,
    other_runs: list[tuple[str, str]] | None = None,
    banner: str | None = None,
) -> str:
    lines = [BEGIN, ""]
    if banner:
        lines += [f"> **{banner}**", ""]
    lines += [
        f"## Verdict: {result.verdict}",
        "",
        f"- Run: `{scorecard['run_id']}` (experiment `{scorecard['exp']}`)",
        f"- Code: commit `{scorecard['commit_sha']}`, config `{scorecard['config_hash'][:19]}`",
        f"- Pre-registration lock: `{prereg_hash or 'none'}`, L1 round: {l1_round or 'n/a'}",
        f"- Subjects usable / requested: {scorecard['n_subjects_usable']} / "
        f"{scorecard['n_subjects_requested']}",
        f"- GPU hours: {scorecard['gpu_hours']:.2f}, computed {computed_at}",
        "",
        "| Gate | Expression | Result |",
        "|---|---|---|",
    ]
    for name, gate in result.gates.items():
        lines.append(f"| {name} | `{gate['expression']}` | {str(gate['result']).lower()} |")
    lines += ["| REVISE | everything else | " + str(result.verdict == "REVISE").lower() + " |", ""]
    lines += ["| Scorecard field | Value |", "|---|---|"]
    lines += [f"| `{key}` | {_format(value)} |" for key, value in result.values.items()]
    lines.append("")
    lines += [f"- Note: {note}" for note in result.notes]
    if other_runs:
        lines.append("Other valid runs of this round (listed so a run cannot be picked silently):")
        lines += [f"- `{run_id}` would give **{verdict}**" for run_id, verdict in other_runs]
    lines += ["", END]
    return "\n".join(lines)


def update_verdict_md(path: Path, block: str, title: str) -> None:
    """Replace only the automatic block; keep everything else exactly as it was."""
    path = Path(path)
    if not path.exists():
        text = f"# Verdict: {title}\n\n{block}\n\n{INTERPRETATION}\n{DEFAULT_INTERPRETATION}"
        write_text_atomic(path, text)
        return
    text = path.read_text(encoding="utf-8")
    split = text.find(INTERPRETATION)
    head, tail = (text, "") if split < 0 else (text[:split], text[split:])
    start, stop = head.find(BEGIN), head.find(END)
    if start >= 0 and stop > start:
        head = head[:start] + block + head[stop + len(END) :]
    elif start >= 0 or stop >= 0:
        raise VerdictError(f"{path}: the automatic verdict markers are damaged; fix them by hand")
    elif tail:
        head = f"{head.rstrip()}\n\n{block}\n\n" if head.strip() else f"{block}\n\n"
    else:
        # No markers at all: treat the whole existing file as human text and keep it.
        head, tail = f"{block}\n\n", f"{INTERPRETATION}\n\n{text}"
    write_text_atomic(path, head + tail)
