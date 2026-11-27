"""L0: the whole kernel on CPU with a synthetic release. Must pass before any GPU is spent."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

for module in ("torch", "nibabel", "pyarrow", "scipy"):
    pytest.importorskip(module)

from labcore.testing import (  # noqa: E402
    experiment_params,
    l0_problems,
    l0_scorecard,
    load_module,
    load_package,
    run_l0,
)
from labcore.verdict.verdict import compute_verdict  # noqa: E402

DIRECTION = Path(__file__).resolve().parents[1]
fake = load_module(DIRECTION / "tests" / "fake_data.py")


def smoke(tmp_path, n_cases=6, **dataset):
    release = fake.make_dataset(tmp_path / "release", n_cases, **dataset)
    base = experiment_params(DIRECTION, "exp_001_pilot")
    params = fake.smoke_params(base, release, tmp_path / "cache", n_cases)
    return run_l0(DIRECTION, params, tmp_path / "l0")


def test_l0_smoke(tmp_path):
    run = smoke(tmp_path, nonfinite=(2,))

    assert run.status["ok"], f"{run.status['traceback_tail']}\n{run.log}"
    for name in run.direction["outputs"]:
        assert (run.out_dir / name).is_file(), name
    assert l0_problems(run) == []

    scorecard = l0_scorecard(run)  # raises if it does not match the schema
    assert scorecard["n_subjects_requested"] == 6
    assert scorecard["n_subjects_usable"] == 6
    assert scorecard["gpu_hours"] == 0 or run.env["gpu_name"]
    assert compute_verdict(run.direction["gates"], scorecard).verdict in {"GO", "REVISE", "PARK"}

    rows = pd.read_csv(run.out_dir / "per_patient.csv")
    assert len(rows) == 6 * 4  # every patient is evaluated once per arm
    assert sorted(rows["fold"].unique()) == [0, 1, 2]
    assert np.isfinite(rows[["dice", "gt_ml", "pred_ml", "abs_vol_err_ml"]].to_numpy()).all()

    cases = pd.read_csv(run.out_dir / "cases.csv").set_index("subject_id")
    damaged = cases.loc["sub-fake0002"]
    assert (damaged["nonfinite_ctp"], damaged["nonfinite_cbf"], damaged["nonfinite_tmax"]) == (3, 2, 1)
    assert (damaged["nonfinite_cbv"], damaged["nonfinite_mtt"], damaged["nonfinite_voxels"]) == (0, 0, 6)
    assert (cases["nonfinite_voxels"].drop("sub-fake0002") == 0).all()
    assert cases["block"].nunique() > 1


def test_an_empty_mask_patient_is_left_out_of_the_statistics(tmp_path):
    run = smoke(tmp_path, empty_lesion=(1,))

    assert run.status["ok"], f"{run.status['traceback_tail']}\n{run.log}"
    assert l0_problems(run) == []
    rows = pd.read_csv(run.out_dir / "per_patient.csv")
    assert set(rows.loc[rows["gt_empty"], "subject_id"]) == {"sub-fake0001"}
    assert len(rows) == 6 * 4  # still trained and evaluated
    scorecard = l0_scorecard(run)
    assert (scorecard["metrics"]["n_primary"], scorecard["metrics"]["n_excluded_empty_gt"]) == (5, 1)
    paired = pd.read_csv(run.out_dir / "paired_stats.csv")
    assert set(paired["n"]) == {5}
    assert "mde149_ucl(raw-maps)" in scorecard["metrics"]


def test_a_run_with_one_mismatched_subject_is_still_valid(tmp_path):
    """Allowed up to 20%: its row must not add columns that results_columns refuses."""
    run = smoke(tmp_path, n_cases=8, shifted=(1,))

    assert run.status["ok"], f"{run.status['traceback_tail']}\n{run.log}"
    assert l0_problems(run) == []
    cases = pd.read_csv(run.out_dir / "cases.csv").set_index("subject_id")
    assert cases.loc["sub-fake0001", "status"].startswith("grid mismatch ['cta']")
    assert l0_scorecard(run)["n_subjects_usable"] == 7


def test_too_many_empty_masks_abort_before_training(tmp_path):
    run = smoke(tmp_path, empty_lesion=(0, 1, 2))

    assert run.status["ok"] is False
    assert "3/6 usable cases have an empty lesion mask" in run.status["error_message"]
    assert not (run.out_dir / "per_patient.csv").exists()


def test_grid_mismatch_stops_before_training(tmp_path):
    run = smoke(tmp_path, grid_mismatch=(0, 1, 2))

    assert run.status["ok"] is False
    assert run.status["stage"] == "run"
    assert "grid mismatch in " in run.status["error_message"]
    assert not (run.out_dir / "per_patient.csv").exists()
    cases = pd.read_csv(run.out_dir / "cases.csv")
    data = importlib.import_module(f"{load_package(DIRECTION / 'src')}.data")
    assert len(cases) == 6 and list(cases.columns) == data.CASE_COLUMNS
    assert cases["status"].str.startswith("grid mismatch").sum() >= 2
