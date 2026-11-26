"""L0: the whole kernel on CPU with synthetic data. Must pass before any GPU is spent."""

from pathlib import Path

from labcore.testing import experiment_params, l0_problems, l0_scorecard, load_module, run_l0

DIRECTION = Path(__file__).resolve().parents[1]


def test_l0_smoke(tmp_path):
    fake = load_module(DIRECTION / "tests" / "fake_data.py")
    data_dir = fake.make_dataset(tmp_path / "data", n_subjects=6)
    params = {
        **experiment_params(DIRECTION, "exp_001"),
        "data_dir": str(data_dir),
        "n_subjects": 6,
        "n_boot": 200,
    }
    run = run_l0(DIRECTION, params, tmp_path / "l0")

    assert run.status["ok"], run.log
    assert l0_problems(run) == []
    scorecard = l0_scorecard(run)
    assert scorecard["n_subjects_usable"] == 6
