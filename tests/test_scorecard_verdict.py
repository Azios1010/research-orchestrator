import json
import math

import pandas as pd
import pytest

from labcore.stats.scorecard import (
    ScorecardError,
    build_scorecard,
    flatten,
    read_source,
    scorecard_problems,
)
from labcore.verdict.verdict import (
    BEGIN,
    END,
    INTERPRETATION,
    VerdictError,
    compute_verdict,
    render_block,
    update_verdict_md,
)

META = {"slug": "s", "exp": "e", "run_id": "r1", "commit_sha": "abc", "config_hash": "sha256:00"}
MAP = {
    "n_subjects_requested": {"file": "summary.json", "path": "n_requested"},
    "n_subjects_usable": {"file": "rows.csv", "column": "dice", "where": {"arm": "raw"}, "agg": "count"},
    "metrics": {
        "mean_dice(raw)": {"file": "rows.csv", "column": "dice", "where": {"arm": "raw"}, "agg": "mean"},
        "ci_lo(raw-maps)": {"file": "summary.json", "path": "contrasts.raw-maps.ci_lo"},
    },
}


@pytest.fixture
def results(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"n_requested": 2, "contrasts": {"raw-maps": {"ci_lo": 0.02}}})
    )
    pd.DataFrame({"arm": ["raw", "raw", "maps"], "dice": [0.4, 0.6, 0.1]}).to_csv(tmp_path / "rows.csv")
    (tmp_path / "status.json").write_text(
        json.dumps({"started_utc": "2026-01-01T00:00:00Z", "finished_utc": "2026-01-01T01:30:00Z"})
    )
    (tmp_path / "env.json").write_text(json.dumps({"gpu_name": "Tesla T4"}))
    return tmp_path


def test_scorecard_from_json_and_csv(results):
    card = build_scorecard(results, MAP, META, pipeline_ok=True)
    assert scorecard_problems(card) == []
    assert card["n_subjects_requested"] == 2 and card["n_subjects_usable"] == 2
    assert card["metrics"] == {"mean_dice(raw)": 0.5, "ci_lo(raw-maps)": 0.02}
    assert card["runtime_min"] == 90 and card["gpu_hours"] == 1.5
    assert flatten(card)["pipeline_ok"] is True


def test_cpu_runs_use_no_gpu_hours(results):
    (results / "env.json").write_text(json.dumps({"gpu_name": None}))
    assert build_scorecard(results, MAP, META, pipeline_ok=True)["gpu_hours"] == 0


@pytest.mark.parametrize(
    "source",
    [
        {"file": "absent.json", "path": "x"},
        {"file": "../outside.json", "path": "x"},
        {"file": "summary.json", "path": "contrasts.raw-N1.ci_lo"},
        {"file": "rows.csv", "column": "nope"},
        {"file": "rows.csv", "column": "dice"},  # three rows and no aggregation
        {"file": "rows.csv", "column": "dice", "where": {"arm": "N9"}, "agg": "mean"},
    ],
)
def test_unreadable_sources_are_errors(results, source):
    with pytest.raises(ScorecardError):
        read_source(results, source)


@pytest.mark.parametrize("bad", [math.nan, math.inf, None])
def test_nan_or_missing_metrics_are_reported(results, bad):
    card = build_scorecard(results, MAP, META, pipeline_ok=True)
    card["metrics"]["ci_lo(raw-maps)"] = bad
    assert any("ci_lo(raw-maps)" in problem for problem in scorecard_problems(card))


def card(**metrics):
    return {
        **META, "n_subjects_requested": 40, "n_subjects_usable": 38, "pipeline_ok": True,
        "gpu_hours": 1.0, "runtime_min": 60.0, "metrics": {"lo": 0.1, "hi": 0.2, **metrics},
    }


GATES = {"GO": "lo > 0", "PARK": "hi <= 0"}


@pytest.mark.parametrize("lo, hi, verdict", [(0.1, 0.2, "GO"), (-0.2, -0.1, "PARK"), (-0.1, 0.1, "REVISE")])
def test_verdicts(lo, hi, verdict):
    assert compute_verdict(GATES, card(lo=lo, hi=hi)).verdict == verdict


def test_conflicting_gates_give_revise_with_a_note():
    result = compute_verdict({"GO": "lo > 0", "PARK": "hi > 0"}, card())
    assert result.verdict == "REVISE" and result.notes


def test_an_invalid_run_never_gets_a_verdict():
    with pytest.raises(VerdictError):
        compute_verdict(GATES, {**card(), "pipeline_ok": False})


def block(verdict_card):
    return render_block(
        compute_verdict(GATES, verdict_card), scorecard=verdict_card, prereg_hash="sha256:x",
        l1_round=1, computed_at="2026-01-01T00:00:00Z",
    )


def test_verdict_md_rewrites_only_the_automatic_block(tmp_path):
    path = tmp_path / "VERDICT.md"
    update_verdict_md(path, block(card()), "T")
    text = path.read_text(encoding="utf-8")
    assert "## Verdict: GO" in text
    assert text.index(BEGIN) < text.index(END) < text.index(INTERPRETATION)

    human = f"\nMy reading. It even quotes {BEGIN} and {END}.\n"
    path.write_text(text.split(INTERPRETATION)[0] + INTERPRETATION + human, encoding="utf-8")
    update_verdict_md(path, block(card(lo=-1.0, hi=-0.5)), "T")
    updated = path.read_text(encoding="utf-8")
    assert "## Verdict: PARK" in updated and "## Verdict: GO" not in updated
    assert updated.endswith(INTERPRETATION + human)


def test_verdict_md_keeps_text_when_markers_are_absent(tmp_path):
    path = tmp_path / "VERDICT.md"
    path.write_text("Old hand-written notes.\n", encoding="utf-8")
    update_verdict_md(path, block(card()), "T")
    text = path.read_text(encoding="utf-8")
    assert text.index(BEGIN) < text.index(INTERPRETATION) < text.index("Old hand-written notes.")


def test_damaged_markers_are_not_overwritten(tmp_path):
    path = tmp_path / "VERDICT.md"
    original = f"{BEGIN}\nhalf a block\n{INTERPRETATION}\nkeep me\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(VerdictError):
        update_verdict_md(path, block(card()), "T")
    assert path.read_text(encoding="utf-8") == original
