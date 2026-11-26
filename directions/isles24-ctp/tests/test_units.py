"""Fast checks that the port kept the notebook's science: arms, folds, statistics, sanity checks."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

for module in ("torch", "nibabel", "pyarrow", "scipy"):
    pytest.importorskip(module)

from labcore.testing import experiment_params, load_module, load_package  # noqa: E402

DIRECTION = Path(__file__).resolve().parents[1]
PACKAGE = load_package(DIRECTION / "src")
arms = importlib.import_module(f"{PACKAGE}.arms")
data = importlib.import_module(f"{PACKAGE}.data")
evaluate = importlib.import_module(f"{PACKAGE}.evaluate")
main = importlib.import_module(f"{PACKAGE}.main")
checks = load_module(DIRECTION / "checks.py")
fake = load_module(DIRECTION / "tests" / "fake_data.py")

CONFIG = experiment_params(DIRECTION, "exp_001_pilot")


def test_pilot_config_is_the_notebook_defaults():
    assert CONFIG == {
        "repo_id": "hugging-science/isles24-stroke", "n_total_shards": 149, "n_cases": 40, "seed": 2027,
        "t_window": 16, "t_pre": 5, "n_baseline": 3, "img_size": 128, "min_brain_frac": 0.02,
        "n_folds": 5, "epochs": 40, "batch_size": 16, "lr": 0.001, "base_ch": 32,
        "lesion_slice_frac": 0.5, "n_boot": 10000, "arms": ["maps", "raw", "N1", "N2"],
    }
    loaded = main.load_params(str(DIRECTION / "configs" / "exp_001_pilot.yaml"))
    assert {k: loaded[k] for k in CONFIG} == CONFIG


def test_unknown_parameters_are_errors(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("params: {epoch: 1}\n")
    with pytest.raises(ValueError, match="epoch"):
        main.load_params(str(path))


def test_chosen_shards_match_the_notebook():
    expected = sorted(np.random.default_rng(2027).choice(149, size=40, replace=False).tolist())
    assert data.choose_shards(CONFIG, np.random.default_rng(2027)) == expected
    assert len(set(expected)) == 40


def patient(seed):
    rng = np.random.default_rng(seed)
    return {
        "ncct": rng.random((3, 8, 8)).astype(np.float16),
        "cta": rng.random((3, 8, 8)).astype(np.float16),
        "maps": rng.random((4, 3, 8, 8)).astype(np.float16),
        "ctp": rng.random((16, 3, 8, 8)).astype(np.float16),
    }


def test_arm_channels():
    d, perm = patient(0), np.random.default_rng(1).permutation(16)
    inputs = {arm: arms.arm_input(arm, d, perm) for arm in arms.ARMS}
    for arm, x in inputs.items():
        assert x.shape == (3, arms.n_channels(arm, 16), 8, 8), arm
        np.testing.assert_array_equal(x[:, 0], d["ncct"])
        np.testing.assert_array_equal(x[:, 1], d["cta"])
    np.testing.assert_array_equal(inputs["maps"][:, 2:], np.transpose(d["maps"], (1, 0, 2, 3)))
    np.testing.assert_array_equal(inputs["raw"][:, 2:], np.transpose(d["ctp"], (1, 0, 2, 3)))
    np.testing.assert_array_equal(inputs["N2"][:, 2:], inputs["raw"][:, 2:][:, perm])
    np.testing.assert_allclose(inputs["N1"][:, 2], d["ctp"].astype(np.float32).mean(0), atol=1e-3)
    np.testing.assert_array_equal(inputs["N1"][:, 3], d["ctp"].max(0))
    with pytest.raises(ValueError):
        arms.arm_input("N9", d, perm)


def test_folds_and_permutations_are_patient_level_and_seeded():
    sids = [f"sub-{i:02d}" for i in range(23)]
    folds = arms.make_folds(sids, 2027, 5)
    assert sorted(s for fold in folds for s in fold) == sids
    assert {len(f) for f in folds} == {4, 5}
    order = np.random.default_rng(2029).permutation(23)
    assert folds[1] == [sids[i] for i in order[1::5]]
    perms = arms.n2_permutations(sids, 2027, 16)
    reference = np.random.default_rng(2028)
    for sid in sids:
        np.testing.assert_array_equal(perms[sid], reference.permutation(16))


def result_rows(n=12, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for arm in arms.ARMS:
            rows.append(
                {
                    "subject_id": f"sub-{i:02d}", "arm": arm, "dice": float(rng.random()),
                    "gt_ml": [1.0, 10.0, 80.0][i % 3], "abs_vol_err_ml": float(rng.random()),
                    "fold": i % 3,
                }
            )
    return pd.DataFrame(rows)


def test_statistics_equal_the_notebook_computation():
    res = result_rows()
    table, exact = evaluate.paired_statistics(res, {**CONFIG, "n_boot": 2000})

    # the notebook's cell, verbatim in substance
    wide = res.pivot_table(index="subject_id", columns="arm", values="dice")
    boot_rng = np.random.default_rng(CONFIG["seed"] + 3)
    expected = []
    for a, b in [("raw", "maps"), ("raw", "N1"), ("raw", "N2"), ("maps", "N1")]:
        d = (wide[a] - wide[b]).dropna().to_numpy()
        means = d[boot_rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        expected.append(
            {
                "contrast": f"{a} - {b}", "n": len(d), "mean_delta": d.mean(), "ci95_lo": lo, "ci95_hi": hi,
                "sd_delta": d.std(ddof=1), "share_patients_a_better": (d > 0).mean(),
                "MDE_at_n149 (2.8*sd/sqrt149)": 2.8 * d.std(ddof=1) / np.sqrt(149),
            }
        )
    expected = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(table, expected.round(4), check_dtype=False)
    assert exact["raw-maps"]["ci_lo"] == pytest.approx(expected["ci95_lo"][0], abs=1e-15)
    assert exact["raw-maps"]["mde149"] == pytest.approx(
        expected["MDE_at_n149 (2.8*sd/sqrt149)"][0], abs=1e-15
    )


def test_lesion_strata():
    table = evaluate.lesion_strata(result_rows())
    assert list(table["n"]) == [4, 4, 4]
    assert list(table["lesion_size"].astype(str)) == ["<5 mL", "5-30 mL", ">30 mL"]


def write_results(folder, requested=10, statuses=None, rows=None):
    statuses = statuses or ["ok"] * requested
    usable = [i for i, s in enumerate(statuses) if s == "ok"]
    if rows is None:
        rows = pd.DataFrame(
            [
                {"subject_id": f"sub-{i}", "arm": arm, "fold": n % 3, "dice": 0.5}
                for n, i in enumerate(usable)
                for arm in arms.ARMS
            ]
        )
    rows.to_csv(folder / "per_patient.csv", index=False)
    pd.DataFrame({"status": statuses}).to_csv(folder / "cases.csv", index=False)
    (folder / "summary.json").write_text(
        json.dumps({"config": {"arms": list(arms.ARMS), "n_folds": 3}, "n_requested": requested})
    )
    return folder


def test_checks_pass_at_exactly_80_percent_usable(tmp_path):
    assert checks.check(write_results(tmp_path, statuses=["ok"] * 8 + ["no brain slices"] * 2)) == []


@pytest.mark.parametrize(
    "statuses, expected",
    [
        (["ok"] * 7 + ["no brain slices"] * 3, "below 80%"),
        (["ok"] * 7 + ["download failed: X"] * 3, "mostly download failed"),
        (["ok"] * 7 + ["grid mismatch ['ctp']"] * 3, "grid mismatch in 3/10"),
    ],
)
def test_checks_on_cases(tmp_path, statuses, expected):
    problems = "\n".join(checks.check(write_results(tmp_path, statuses=statuses)))
    assert expected in problems


def test_grid_mismatch_at_exactly_20_percent_is_allowed(tmp_path):
    statuses = ["ok"] * 8 + ["grid mismatch ['ctp']"] * 2
    assert checks.check(write_results(tmp_path, statuses=statuses)) == []


@pytest.mark.parametrize(
    "change, expected",
    [
        (lambda r: r[~((r.arm == "N2") & (r.fold == 1))], "missing results for arm N2, fold 1"),
        (lambda r: r.assign(dice=r.dice.where(r.arm != "raw", 0.0)), "zero for every patient in arm raw"),
        (lambda r: r.assign(dice=r.dice.where(r.index != 0, np.nan)), "missing or outside"),
        (lambda r: r.assign(dice=r.dice.where(r.index != 0, 1.5)), "missing or outside"),
        (lambda r: pd.concat([r, r.iloc[:1]]), "more than once"),
    ],
)
def test_checks_on_rows(tmp_path, change, expected):
    write_results(tmp_path)
    rows = change(pd.read_csv(tmp_path / "per_patient.csv"))
    problems = "\n".join(checks.check(write_results(tmp_path, rows=rows)))
    assert expected in problems


def test_make_finite():
    x = np.array([np.nan, np.inf, -np.inf, -2.0, 5.0], dtype=np.float32)
    out, count = data.make_finite(x)
    assert count == 3
    np.testing.assert_array_equal(out, [0.0, 5.0, -2.0, -2.0, 5.0])
    assert np.isnan(x[0])  # the input is left as it was
    out, count = data.make_finite(np.array([np.nan, np.inf]))
    assert count == 2 and (out == 0).all()
    clean = np.arange(4.0)
    assert data.make_finite(clean) == (clean, 0)


def test_integer_factor_is_a_block_mean():
    x = np.random.default_rng(0).random((2, 12, 8)).astype(np.float32)
    out = data.area_down(x, 4)
    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out[1, 2, 3], x[1, 6:9, 6:8].mean(), rtol=1e-6)
    np.testing.assert_allclose(out.mean(), x.mean(), rtol=1e-6)  # nothing is skipped
    np.testing.assert_array_equal(data.area_down(x[:, :8, :], 8), x[:, :8, :])


def test_mask_needs_half_of_the_block():
    mask = np.zeros((1, 8, 8), dtype=bool)
    mask[0, 0:2, 0:4] = True  # 8 of 16 voxels of block (0, 0)
    mask[0, 0:2, 4:8] = True
    mask[0, 0, 4] = False  # 7 of 16 voxels of block (0, 1)
    mask[0, 4:8, 4:8] = True  # a full block
    out = data.area_down(mask, 2, is_mask=True)
    assert out.dtype == bool
    np.testing.assert_array_equal(out[0], [[True, False], [False, True]])


@pytest.mark.parametrize("shape", [(2, 20, 20), (2, 12, 10), (2, 6, 6)])
def test_other_factors_are_smoothed_then_zoomed(shape):
    constant = np.full(shape, 3.5, dtype=np.float32)
    np.testing.assert_allclose(data.area_down(constant, 8), 3.5, rtol=1e-5)
    assert data.area_down(constant > 0, 8, is_mask=True).all()
    assert not data.area_down(constant < 0, 8, is_mask=True).any()


def test_prefilter_removes_aliasing():
    """Alternating stripes cannot be represented at the lower resolution: they must average out."""
    stripes = np.zeros((1, 40, 40), dtype=np.float32)
    stripes[:, :, ::2] = 1.0
    for size in (10, 16):  # an integer factor and a non-integer one
        out = data.area_down(stripes, size)[0, :, 1:-1]  # the border column depends on edge handling
        assert np.abs(out - 0.5).max() < 0.05, size


def test_every_input_is_downsampled_by_block_mean(tmp_path):
    """Factors of 5 and 6 on fake shards: each stored array equals the block mean of the full one."""
    shards = fake.make_dataset(tmp_path / "shards", 1, nonfinite=(0,))
    (tmp_path / "cache").mkdir()
    cfg = {**CONFIG, "cache_dir": str(tmp_path / "cache"), "img_size": 16}
    arrays, meta = data.preprocess_case(str(shards / "train-00000-of-00001.parquet"), cfg, [])
    assert meta["status"] == "ok" and meta["nonfinite_voxels"] == 6

    case = fake.add_nonfinite(fake.make_case(np.random.default_rng(0)))
    zyx = {name: data.xyz_to_zyx(a.astype(np.float32)) for name, a in case.items()}
    brain = data.brain_mask_from_ncct(zyx["ncct"])
    sl = np.where(brain.reshape(len(brain), -1).mean(1) >= cfg["min_brain_frac"])[0]

    def blocks(a):
        z, h, w = a.shape
        return a.reshape(z, 16, h // 16, 16, w // 16).mean(axis=(2, 4))

    expected = {
        "ncct": blocks((np.clip(zyx["ncct"], 0, 80) / 80.0)[sl]),
        "cta": blocks((np.clip(zyx["cta"], -100, 400) / 500.0 + 0.2)[sl]),
        "mtt": blocks((np.clip(data.make_finite(zyx["mtt"])[0], 0, 20) / 20.0)[sl]),
        "tmax": blocks((np.clip(data.make_finite(zyx["tmax"])[0], 0, 20) / 20.0)[sl]),
    }
    stored = {"ncct": arrays["ncct"], "cta": arrays["cta"]}
    stored.update(mtt=arrays["maps"][2], tmax=arrays["maps"][3])
    for name, want in expected.items():
        np.testing.assert_allclose(stored[name].astype(np.float32), want, atol=1e-3, err_msg=name)
    np.testing.assert_array_equal(arrays["mask"], blocks((zyx["lesion_mask"] > 0.5)[sl] * 1.0) >= 0.5)
    np.testing.assert_array_equal(arrays["brain"], blocks(brain[sl] * 1.0) >= 0.5)

    def frame(t):
        return data.make_finite(data.xyz_to_zyx(case["ctp"][..., t].astype(np.float32)))[0]

    frames = [frame(t) for t in range(case["ctp"].shape[3])]
    base = np.mean(frames[: cfg["n_baseline"]], axis=0)
    start = meta["peak"] - cfg["t_pre"]
    for k, t in enumerate(np.clip(np.arange(start, start + cfg["t_window"]), 0, meta["T"] - 1)):
        want = blocks(np.clip(frames[t] - base, -20, 150)[sl] / 100.0)
        np.testing.assert_allclose(arrays["ctp"][k].astype(np.float32), want, atol=1e-3, err_msg=f"frame {k}")

    for name in ("ncct", "cta", "maps", "ctp"):
        assert np.isfinite(arrays[name]).all(), name


def test_every_array_goes_through_area_down(tmp_path, monkeypatch):
    shards = fake.make_dataset(tmp_path / "shards", 1)
    (tmp_path / "cache").mkdir()
    calls, original = [], data.area_down

    def spy(x, size, is_mask=False):
        calls.append(is_mask)
        return original(x, size, is_mask)

    monkeypatch.setattr(data, "area_down", spy)
    cfg = {**CONFIG, "cache_dir": str(tmp_path / "cache"), "img_size": 64}
    data.preprocess_case(str(shards / "train-00000-of-00001.parquet"), cfg, [])
    # 16 CTP frames, NCCT, CTA and 4 maps as images; brain and lesion as masks
    assert (calls.count(False), calls.count(True)) == (16 + 2 + 4, 2)


def test_cases_table_holds_whole_numbers(tmp_path):
    shards = fake.make_dataset(tmp_path / "shards", 3, nonfinite=(0,), grid_mismatch=(2,))
    (tmp_path / "cache").mkdir()
    cfg = {**CONFIG, "cache_dir": str(tmp_path / "cache"), "local_shard_dir": str(shards),
           "n_total_shards": 3, "img_size": 64}
    _, table = data.load_cases(cfg, [0, 1, 2])
    assert str(table["nonfinite_voxels"].dtype) == "Int64"
    assert table["nonfinite_voxels"].tolist()[:2] == [6, 0]
    assert pd.isna(table["nonfinite_voxels"][2])  # excluded before preprocessing


def test_too_few_cases_names_the_reason():
    cfg = main.load_params(str(DIRECTION / "configs" / "exp_001_pilot.yaml"))
    chosen = list(range(10))
    download = pd.DataFrame({"status": ["ok"] * 3 + ["download failed: ConnectionError: x"] * 7})
    with pytest.raises(RuntimeError, match="mostly download failed"):
        main.check_usable(cfg, chosen, download, 3)
    other = pd.DataFrame({"status": ["ok"] * 3 + ["error ValueError: x"] * 6 + ["download failed: x"]})
    with pytest.raises(RuntimeError) as info:
        main.check_usable(cfg, chosen, other, 3)
    assert "download failed (" not in str(info.value) and "too few usable cases: 3/10" in str(info.value)


def test_resume_only_with_identical_settings(tmp_path):
    assert main.resume_rows(tmp_path, "key-1") == []
    pd.DataFrame([{"arm": "raw", "fold": 0}]).to_csv(tmp_path / "per_patient.csv", index=False)
    assert main.resume_rows(tmp_path, "key-1") == [{"arm": "raw", "fold": 0}]
    assert main.resume_rows(tmp_path, "key-2") == []
