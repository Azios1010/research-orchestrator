"""L0: the whole kernel on CPU with synthetic archives. Must pass before any GPU is spent."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

for module in ("torch", "sklearn", "PIL"):
    pytest.importorskip(module)

from labcore.testing import (  # noqa: E402
    experiment_params,
    l0_problems,
    l0_scorecard,
    load_module,
    run_l0,
)
from labcore.verdict.verdict import compute_verdict  # noqa: E402

DIRECTION = Path(__file__).resolve().parents[1]
fake = load_module(DIRECTION / "tests" / "fake_data.py")
checks = load_module(DIRECTION / "checks.py")


def smoke(work: Path, change=None, **dataset):
    data = fake.make_dataset(work / "data", **dataset)
    base = experiment_params(DIRECTION, "exp_001_pilot")
    params = fake.smoke_params(base, data, work / "cache")
    if change:
        change(params)
    return run_l0(DIRECTION, params, work / "l0")


@pytest.fixture(scope="module")
def positive(tmp_path_factory):
    """Patients differ from each other in every class: seeing a patient's cells must help."""
    return smoke(tmp_path_factory.mktemp("positive"), patient_effect=1.0)


def test_l0_smoke_positive_control(positive):
    run = positive
    assert run.status["ok"], f"{run.status['traceback_tail']}\n{run.log}"
    for name in run.direction["outputs"]:
        assert (run.out_dir / name).is_file(), name
    assert l0_problems(run) == []

    scorecard = l0_scorecard(run)  # raises if it does not match the schema
    assert scorecard["n_subjects_requested"] == scorecard["n_subjects_usable"] == 15
    metrics = scorecard["metrics"]
    assert metrics["ci_lo(shared-disjoint)"] > 0
    assert metrics["mean(shared-disjoint)"] > 0.1
    lo, mean, hi = (metrics[f"{name}(shared-disjoint)"] for name in ("ci_lo", "mean", "ci_hi"))
    assert lo <= mean <= hi
    assert metrics["n_eval_in_train"] == metrics["n_test_patient_cells_in_disjoint_train"] == 0
    assert metrics["n_unconverged_probes"] == 0
    assert metrics["same_patient_share"] > 2 * metrics["same_patient_share_at_random_same_class"]
    assert compute_verdict(run.direction["gates"], scorecard).verdict == "GO"


def test_both_arms_differ_only_by_the_swapped_cells(positive):
    folds = pd.read_csv(positive.out_dir / "folds.csv")
    assert len(folds) == 5 * 3 * 2
    wide = folds.pivot_table(index=["fold", "split_seed"], columns="arm", values=["n_train", "n_eval"])
    assert (wide["n_train"]["shared"] == wide["n_train"]["disjoint"]).all()
    assert (wide["n_eval"]["shared"] == wide["n_eval"]["disjoint"]).all()
    assert (folds.groupby(["fold", "split_seed"])["eval_set_hash"].nunique() == 1).all()
    shared = folds[folds["arm"] == "shared"]
    assert (shared["n_train_donor"] > 0).all()
    cells = pd.read_csv(positive.out_dir / "patient_counts.csv")
    assert sorted(cells["patient_index"].unique()) == list(range(15))


def test_no_patient_or_image_name_leaves_the_kernel(positive):
    texts = [positive.log] + [
        path.read_text(encoding="utf-8") for path in positive.out_dir.iterdir() if path.is_file()
    ]
    for text in texts:
        assert "Patient_" not in text
        assert "Signed slides/" not in text
        assert "00000001.jpg" not in text


@pytest.mark.parametrize(
    "file, change, expected",
    [
        ("folds.csv", lambda t: t.assign(n_eval_in_train=t["n_eval_in_train"] + (t.index == 0)),
         "evaluated cells are in a train set"),
        ("folds.csv", lambda t: t.assign(n_eval_hash_in_train=t["n_eval_hash_in_train"] + (t.index == 1)),
         "images equal to an evaluated one"),
        ("folds.csv",
         lambda t: t.assign(n_test_patient_cells_in_train=t["n_test_patient_cells_in_train"].where(
             t.arm != "disjoint", 1)), "disjoint train set holds cells of test patients"),
        ("folds.csv", lambda t: t.assign(n_train=t["n_train"] + (t.arm == "shared")),
         "the arms differ in train size"),
        ("folds.csv", lambda t: t.assign(eval_set_hash=t["eval_set_hash"].where(t.index != 0, "x")),
         "the arms differ in eval set"),
        ("folds.csv", lambda t: t.assign(probe_converged=t.index != 0), "did not converge"),
        ("folds.csv", lambda t: t.iloc[1:], "each fold x seed x arm once"),
        ("split_counts.csv", lambda t: t.assign(n_train=t["n_train"] + (t.arm == "shared")),
         "train count of some class"),
        ("dataset_counts.csv", lambda t: t.assign(images=t["images"] + (t.index == 0)), "not as expected"),
        ("patient_counts.csv", lambda t: t.assign(tp=t["tp"] + (t.index == 0)),
         "cells scored per class"),
        ("contrasts.csv", lambda t: t.assign(mean_delta=t["mean_delta"] + 1.0), "disagree on the gated"),
    ],
)
def test_checks_catch_a_broken_run(positive, tmp_path, file, change, expected):
    broken = tmp_path / "out"
    shutil.copytree(positive.out_dir, broken)
    change(pd.read_csv(broken / file)).to_csv(broken / file, index=False)
    assert expected in "\n".join(checks.check(broken))


@pytest.mark.parametrize(
    "edit, expected",
    [
        (lambda s: s.update(n_usable=11), "usable patients 11/15 is below 80%"),
        (lambda s: s["data"].update(source="download"), "must use DinoBloom-S"),
        (lambda s: s["data"].update(conflicting_groups=3), "conflicting classes"),
        (lambda s: s["arms"]["shared"].update(macro_f1_primary=0.0), "zero in arm shared"),
    ],
)
def test_checks_read_the_summary(positive, tmp_path, edit, expected):
    broken = tmp_path / "out"
    shutil.copytree(positive.out_dir, broken)
    summary = json.loads((broken / "summary.json").read_text(encoding="utf-8"))
    edit(summary)
    (broken / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert expected in "\n".join(checks.check(broken))


def test_usable_at_exactly_80_percent_is_allowed(positive, tmp_path):
    result = tmp_path / "out"
    shutil.copytree(positive.out_dir, result)
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    summary["n_usable"] = 12
    (result / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert checks.check(result) == []


def test_negative_control_shows_no_gain(tmp_path):
    """No patient-specific signal: swapping cells in cannot help."""
    run = smoke(tmp_path, patient_effect=0.0)
    assert run.status["ok"], f"{run.status['traceback_tail']}\n{run.log}"
    assert l0_problems(run) == []
    metrics = l0_scorecard(run)["metrics"]
    assert abs(metrics["mean(shared-disjoint)"]) < 0.03
    assert metrics["ci_lo(shared-disjoint)"] <= 0.02
    assert metrics["macro_f1(disjoint)"] > 0.3  # the class signal itself is learnt
    assert compute_verdict(run.direction["gates"], l0_scorecard(run)).verdict != "GO"


def test_unexpected_counts_stop_the_run_before_features(tmp_path):
    def change(params):
        params["expected_deepapl"] = {**params["expected_deepapl"], "Basophil": [1, 1]}

    run = smoke(tmp_path, change=change)
    assert run.status["ok"] is False
    assert "the data differ from the expected counts" in run.status["error_message"]
    assert "'Basophil'" in run.status["error_message"]
    assert (run.out_dir / "dataset_counts.csv").is_file()
    assert not (run.out_dir / "folds.csv").exists()
    assert "features" not in run.log
