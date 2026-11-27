"""Fast checks: data access, preprocessing, arms, folds, statistics, sanity checks and gates."""

import importlib
import json
import re
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

for module in ("torch", "nibabel", "pyarrow", "scipy"):
    pytest.importorskip(module)

from scipy.stats import chi2  # noqa: E402

from labcore.testing import experiment_params, load_module, load_package  # noqa: E402
from labcore.watcher.classify import classify  # noqa: E402

DIRECTION = Path(__file__).resolve().parents[1]
PACKAGE = load_package(DIRECTION / "src")
arms = importlib.import_module(f"{PACKAGE}.arms")
data = importlib.import_module(f"{PACKAGE}.data")
evaluate = importlib.import_module(f"{PACKAGE}.evaluate")
main = importlib.import_module(f"{PACKAGE}.main")
zenodo = importlib.import_module(f"{PACKAGE}.zenodo")
checks = load_module(DIRECTION / "checks.py")
fake = load_module(DIRECTION / "tests" / "fake_data.py")

CONFIG = experiment_params(DIRECTION, "exp_001_pilot")
MANIFEST = zenodo.load_manifest()
# The 40 subjects of the pilot, pre-registered: sub-stroke<number>
PILOT = [
    1, 2, 9, 10, 15, 16, 22, 25, 36, 37, 45, 47, 55, 57, 73, 74, 80, 81, 86, 87,
    93, 94, 99, 100, 106, 107, 112, 113, 118, 119, 138, 139, 146, 147, 154, 155, 164, 172, 181, 188,
]


def test_pilot_config_keeps_the_analysis_settings():
    assert CONFIG == {
        "repo_id": "hugging-science/isles24-stroke", "n_total_shards": 149, "n_cases": 40, "seed": 2027,
        "t_window": 16, "t_pre": 5, "n_baseline": 3, "img_size": 128, "min_brain_frac": 0.02,
        "n_folds": 5, "epochs": 40, "batch_size": 16, "lr": 0.001, "base_ch": 32,
        "lesion_slice_frac": 0.5, "n_boot": 10000, "arms": ["maps", "raw", "N1", "N2"],
        "download_threads": 22, "prepare_threads": 2, "probe_minutes": 10, "max_fetch_hours": 2.5,
        "min_free_gb": 30,
    }
    loaded = main.load_params(str(DIRECTION / "configs" / "exp_001_pilot.yaml"))
    assert {k: loaded[k] for k in CONFIG} == CONFIG
    assert loaded["archive"] == loaded["manifest"] == loaded["local_shard_dir"] == ""


def test_unknown_parameters_are_errors(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("params: {epoch: 1}\n")
    with pytest.raises(ValueError, match="epoch"):
        main.load_params(str(path))


# ---------------------------------------------------------------- the manifest and the selection


def test_the_manifest_covers_the_release():
    assert MANIFEST["url"].endswith("/records/17652035/files/train.7z/content")
    assert len(MANIFEST["subjects"]) == 149
    assert all(set(files) == set(zenodo.KINDS) for files in MANIFEST["subjects"].values())
    blocks = sorted(MANIFEST["blocks"].values(), key=lambda b: b["offset"])
    assert all(a["offset"] + a["packed"] <= b["offset"] for a, b in zip(blocks, blocks[1:], strict=False))
    assert blocks[-1]["offset"] + blocks[-1]["packed"] <= MANIFEST["archive_size"]
    for files in zenodo.plan(MANIFEST, sorted(MANIFEST["subjects"])).values():
        assert all(a[0] + a[1] <= b[0] for a, b in zip(files, files[1:], strict=False))


def test_the_pilot_subjects_are_the_registered_ones():
    chosen = zenodo.select_subjects(MANIFEST, 40)
    assert sorted(chosen) == [f"sub-stroke{n:04d}" for n in PILOT]
    assert chosen == zenodo.select_subjects(MANIFEST, 44)[:40]
    wanted = zenodo.plan(MANIFEST, chosen)
    assert len(wanted) == 22  # spread over every block that holds registered files
    assert max(len({entry[3] for entry in files}) for files in wanted.values()) == 2
    assert 30e9 < zenodo.expected_bytes(wanted) < 35e9  # of about 99 GB
    assert sorted(zenodo.select_subjects(MANIFEST, 149)) == sorted(MANIFEST["subjects"])
    for n in (0, 150):
        with pytest.raises(ValueError):
            zenodo.select_subjects(MANIFEST, n)


def test_shard_numbers_match_the_mirror():
    # FACT: pairs read from cases.csv of run isles24-ctp-exp-001-pilot-20261127t042526z-a860b3
    seen = {1: 0, 2: 1, 4: 3, 10: 9, 135: 95, 176: 135, 184: 143}
    for number, shard in seen.items():
        assert data.shard_index(MANIFEST, f"sub-stroke{number:04d}") == shard


# ---------------------------------------------------------------- streaming a block


def small_archive(tmp_path, n=3):
    rng = np.random.default_rng(5)
    payloads = {
        (f"s{i}", kind): bytes(rng.integers(0, 256, int(rng.integers(500, 4000)), dtype=np.uint8))
        for i in range(n)
        for kind in zenodo.KINDS
    }
    files = [(subject, kind, payload) for (subject, kind), payload in payloads.items()]
    manifest = fake.write_archive(tmp_path / "a.bin", files, per_block=len(files))
    wanted = zenodo.plan(manifest, [f"s{n - 1}", "s0"])[0]
    return (tmp_path / "a.bin").read_bytes(), manifest["blocks"]["0"], wanted, payloads


def extract(open_range, block, wanted, folder, **options):
    done = []
    options.setdefault("stop", threading.Event())
    options.setdefault("sleep", lambda seconds: None)
    zenodo.stream_block(
        open_range, block, wanted, folder, lambda s, k, p: done.append((s, k, p)), **options
    )
    return done


def test_only_the_wanted_files_are_written(tmp_path):
    _, block, wanted, payloads = small_archive(tmp_path)
    (tmp_path / "out").mkdir()
    done = extract(zenodo.open_file(tmp_path / "a.bin"), block, wanted, tmp_path / "out")
    assert [(s, k) for s, k, _ in done] == [(entry[3], entry[4]) for entry in wanted]
    assert all(path.read_bytes() == payloads[s, k] for s, k, path in done)
    assert len(list((tmp_path / "out").iterdir())) == 2 * len(zenodo.KINDS)


def test_a_dropped_connection_resumes_where_it_stopped(tmp_path, monkeypatch):
    content, block, wanted, payloads = small_archive(tmp_path)
    monkeypatch.setattr(zenodo, "CHUNK", 700)
    starts = []

    class Drops:
        def __init__(self, start, end):
            self.data, self.pos, self.reads = content[start : end + 1], 0, 0

        def read(self, n):
            self.reads += 1
            if self.reads > 3:
                raise ConnectionResetError("dropped")
            piece = self.data[self.pos : self.pos + n]
            self.pos += len(piece)
            return piece

        def close(self):
            pass

    def opener(start, end):
        starts.append(start)
        return Drops(start, end)

    done = extract(opener, block, wanted, tmp_path, retries=1)  # progress resets the failure count
    assert all(path.read_bytes() == payloads[s, k] for s, k, path in done)
    assert len(starts) > 3
    assert [s - block["offset"] for s in starts] == [2100 * i for i in range(len(starts))]


@pytest.mark.parametrize(
    ("error", "calls", "message"),
    [
        (urllib.error.URLError("no route"), 3, "gave up after 3 attempts"),
        (urllib.error.HTTPError("u", 429, "slow down", {}, None), 3, "gave up after 3 attempts"),
        (urllib.error.HTTPError("u", 404, "gone", {}, None), 1, "HTTP 404"),
    ],
)
def test_retries_and_refusals(tmp_path, error, calls, message):
    _, block, wanted, _ = small_archive(tmp_path)
    seen = []

    def opener(start, end):
        seen.append(start)
        raise error

    with pytest.raises(zenodo.SourceError, match=message):
        extract(opener, block, wanted, tmp_path, retries=2)
    assert len(seen) == calls


def test_wrong_bytes_are_refused(tmp_path):
    _, block, wanted, _ = small_archive(tmp_path)
    start, size, crc, subject, kind = wanted[1]
    wanted[1] = (start, size, crc ^ 1, subject, kind)
    with pytest.raises(zenodo.SourceError, match=f"CRC mismatch in {subject} {kind}"):
        extract(zenodo.open_file(tmp_path / "a.bin"), block, wanted, tmp_path)


def test_a_stop_request_ends_the_stream(tmp_path):
    _, block, wanted, _ = small_archive(tmp_path)
    stop = threading.Event()
    stop.set()
    with pytest.raises(zenodo.Stopped):
        extract(zenodo.open_file(tmp_path / "a.bin"), block, wanted, tmp_path, stop=stop)


def test_over_http_byte_ranges_are_required(tmp_path):
    content, block, wanted, payloads = small_archive(tmp_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            if self.path == "/ranges":
                start, end = int(match[1]), int(match[2])
                body = content[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            else:
                body = content
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        done = extract(zenodo.open_url(f"{url}/ranges"), block, wanted, tmp_path)
        assert all(path.read_bytes() == payloads[s, k] for s, k, path in done)
        with pytest.raises(zenodo.SourceError, match="did not honour a byte range"):
            extract(zenodo.open_url(f"{url}/whole"), block, wanted, tmp_path)
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- stopping early


def reason_for(statuses=(), received=33e9, elapsed=100.0):
    cfg = {"max_grid_mismatch_frac": 0.2, "max_fetch_hours": 2.5, "probe_minutes": 10}
    return data.stop_reason(cfg, list(statuses), 40, received, 33e9, elapsed)


def failure_class(message):
    status = {"error_type": "RuntimeError", "error_message": message}
    return classify(run_status="ERROR", status_json=status).failure_class


def test_nothing_stops_a_healthy_fetch():
    assert reason_for(["ok"] * 10, received=3e9, elapsed=600) is None  # 5 MB/s: under two hours
    assert reason_for(received=1e6, elapsed=599) is None  # too early to judge the speed
    assert reason_for(["ok"] * 32 + ["download failed: x"] * 8) is None  # 20% may still be valid


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"received": 1.5e8, "elapsed": 600}, "too slow: 0.25 MB/s"),
        ({"elapsed": 9001}, "passed its limit of 2.5 h"),
        ({"statuses": ["grid mismatch ['ctp']: x"] * 9}, "grid mismatch in 9/40"),
        ({"statuses": ["source mismatch: x"] * 9}, "9/40 cases unusable"),
    ],
)
def test_reasons_that_are_never_retried(change, message):
    reason = reason_for(**change)
    assert message in reason
    assert failure_class(reason) == "code"  # a retry would spend the GPU hours again


def test_mostly_failed_downloads_may_be_retried():
    reason = reason_for(["download failed: SourceError: gave up after 9 attempts"] * 9)
    assert "mostly download failed (9 of 9 failures)" in reason
    assert failure_class(reason) == "infra"


# ---------------------------------------------------------------- loading cases


def load(tmp_path, monkeypatch, n=6, img_size=16, damage=(), dataset=None, **options):
    monkeypatch.setattr(data, "POLL_SECONDS", 0.02)
    release = fake.make_dataset(tmp_path / "release", n, **(dataset or {}))
    manifest = zenodo.load_manifest(release["manifest"])
    content = bytearray(release["archive"].read_bytes())
    for number in damage:  # make a block undecodable
        offset = manifest["blocks"][str(number)]["offset"]
        content[offset : offset + 20] = b"\x7f" * 20
    release["archive"].write_bytes(bytes(content))
    params = fake.smoke_params(CONFIG, release, tmp_path / "cache", n)
    cfg = {**main.DEFAULTS, **params, "img_size": img_size, **options}
    return data.load_cases(cfg, manifest, zenodo.select_subjects(manifest, n))


def test_cases_are_loaded_and_the_cache_is_emptied(tmp_path, monkeypatch):
    cases, table, stopped = load(tmp_path, monkeypatch, dataset={"nonfinite": (0,)})
    assert stopped is None and len(cases) == 6 and (table["status"] == "ok").all()
    assert list(table.columns) == data.CASE_COLUMNS
    assert str(table["nonfinite_voxels"].dtype) == "Int64"
    assert sorted(table["shard"]) == list(range(6))
    assert list((tmp_path / "cache").iterdir()) == []
    arrays, meta = cases["sub-fake0000"]
    assert arrays["ctp"].shape == (16, meta["n_slices"], 16, 16)
    assert arrays["mask"].any() and arrays["brain"].any()


def test_results_columns_list_the_case_table():
    from labcore.config import load_direction

    assert load_direction(DIRECTION)["results_columns"]["cases.csv"] == data.CASE_COLUMNS


def test_unaligned_or_unmatched_volumes_are_excluded(tmp_path, monkeypatch):
    bad = {"grid_mismatch": (1,), "shifted": (2,), "source_mismatch": (3,)}
    cases, table, stopped = load(tmp_path, monkeypatch, n=16, dataset=bad)
    status = table.set_index("subject_id")["status"]
    assert status["sub-fake0001"].startswith("grid mismatch ['ctp']")
    assert status["sub-fake0002"].startswith("grid mismatch ['cta']")  # same shape, other affine
    assert status["sub-fake0003"].startswith("source mismatch: the lesion mask differs")
    assert stopped is None and len(cases) == 13
    assert list(table.columns) == data.CASE_COLUMNS


def test_one_unreadable_block_loses_only_its_subjects(tmp_path, monkeypatch):
    # eight files per block: one subject each
    cases, table, stopped = load(tmp_path, monkeypatch, damage=(1,), dataset={"per_block": 8})
    status = table.set_index("subject_id")["status"]
    assert status["sub-fake0001"].startswith("download failed: SourceError")
    assert stopped is None and sorted(cases) == [f"sub-fake{i:04d}" for i in (0, 2, 3, 4, 5)]


def test_many_unreadable_blocks_stop_the_run(tmp_path, monkeypatch):
    _, table, stopped = load(tmp_path, monkeypatch, damage=(0, 1, 2), dataset={"per_block": 8})
    assert "mostly download failed" in stopped
    assert len(table) == 6 and list(table.columns) == data.CASE_COLUMNS


def test_a_broken_stream_stops_the_others_at_once(tmp_path, monkeypatch):
    told_to_stop = []

    def stream(open_range, block, files, dest, on_file, *, stop, **options):
        if files[0][3] == "sub-fake0000":
            raise OSError("No space left on device")
        told_to_stop.append(stop.wait(60))
        raise zenodo.Stopped("stopped")

    monkeypatch.setattr(data.zenodo, "stream_block", stream)
    started = time.monotonic()
    with pytest.raises(OSError, match="No space left"):
        load(tmp_path, monkeypatch, dataset={"per_block": 8}, download_threads=6)
    assert all(told_to_stop) and time.monotonic() - started < 50


def test_too_little_disk_stops_before_downloading(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="GB free for the cache"):
        load(tmp_path, monkeypatch, min_free_gb=1e9)


# ---------------------------------------------------------------- arms, folds, statistics


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
                    "gt_empty": False, "gt_ml": [1.0, 10.0, 80.0][i % 3],
                    "abs_vol_err_ml": float(rng.random()), "fold": i % 3,
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
                "MDE_UCL80_at_n149": 2.8 * d.std(ddof=1) * np.sqrt((len(d) - 1) / chi2.ppf(0.20, len(d) - 1))
                / np.sqrt(149),
            }
        )
    expected = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(table, expected.round(4), check_dtype=False)
    assert exact["raw-maps"]["ci_lo"] == pytest.approx(expected["ci95_lo"][0], abs=1e-15)
    assert exact["raw-maps"]["mde149"] == pytest.approx(
        expected["MDE_at_n149 (2.8*sd/sqrt149)"][0], abs=1e-15
    )


def test_empty_mask_subjects_are_left_out_of_every_statistic():
    res = result_rows()
    empty = ["sub-00", "sub-05"]
    res.loc[res.subject_id.isin(empty), "gt_empty"] = True
    res.loc[res.subject_id.isin(empty), "dice"] = 1.0  # what an empty prediction would score
    table, exact = evaluate.paired_statistics(res, {**CONFIG, "n_boot": 500})
    assert set(table["n"]) == {10} and exact["raw-maps"]["n"] == 10
    kept = result_rows()
    kept = kept[~kept.subject_id.isin(empty)]
    config = {**CONFIG, "n_boot": 500}
    same_table, same_exact = evaluate.paired_statistics(kept.assign(gt_empty=False), config)
    pd.testing.assert_frame_equal(table, same_table)
    assert exact == same_exact
    assert evaluate.lesion_strata(res)["n"].sum() == 10


def test_mde_with_the_upper_bound_is_in_the_contrasts():
    _, exact = evaluate.paired_statistics(result_rows(), {**CONFIG, "n_boot": 500})
    stat = exact["raw-maps"]
    expected = 2.8 * stat["sd_delta"] * np.sqrt(11 / chi2.ppf(0.20, 11)) / np.sqrt(149)
    assert stat["mde149_ucl"] == pytest.approx(expected, rel=1e-9)
    assert stat["mde149_ucl"] > stat["mde149"]


def test_a_contrast_needs_two_subjects_with_a_lesion():
    res = result_rows(n=3)
    res.loc[res.subject_id.isin(["sub-00", "sub-01"]), "gt_empty"] = True
    with pytest.raises(ValueError, match="fewer than 2"):
        evaluate.paired_statistics(res, {**CONFIG, "n_boot": 100})


def test_too_many_empty_masks_stop_the_run_before_training():
    def case(empty):
        mask = np.zeros((2, 4, 4), bool)
        if not empty:
            mask[0, 0, 0] = True
        return {"mask": mask}, {}

    cfg = {"max_empty_gt_frac": 0.25}
    main.check_empty_masks(cfg, {f"s{i}": case(i < 1) for i in range(4)})  # exactly 25%: allowed
    with pytest.raises(RuntimeError, match="2/4 usable cases have an empty lesion mask"):
        main.check_empty_masks(cfg, {f"s{i}": case(i < 2) for i in range(4)})


def test_lesion_strata():
    table = evaluate.lesion_strata(result_rows())
    assert list(table["n"]) == [4, 4, 4]
    assert list(table["lesion_size"].astype(str)) == ["<5 mL", "5-30 mL", ">30 mL"]


# ---------------------------------------------------------------- checks on a finished run


def write_results(folder, requested=10, statuses=None, rows=None, excluded_empty=0):
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
    excluded = excluded_empty
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "config": {"arms": list(arms.ARMS), "n_folds": 3}, "n_requested": requested,
                "n_excluded_empty_gt": excluded, "n_primary": len(usable) - excluded,
            }
        )
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


def test_checks_on_empty_masks(tmp_path):
    assert checks.check(write_results(tmp_path, requested=12, excluded_empty=3)) == []  # exactly 25%
    problems = "\n".join(checks.check(write_results(tmp_path, requested=12, excluded_empty=4)))
    assert "4/12 usable cases have an empty lesion mask" in problems
    folder = write_results(tmp_path, requested=12, excluded_empty=1)
    summary = json.loads((folder / "summary.json").read_text())
    summary["n_primary"] = 5
    (folder / "summary.json").write_text(json.dumps(summary))
    assert "is not the 12 usable" in "\n".join(checks.check(folder))
    summary.update(n_primary=1, n_excluded_empty_gt=11)
    (folder / "summary.json").write_text(json.dumps(summary))
    assert "only 1 subject(s) in the paired statistics" in "\n".join(checks.check(folder))


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


# ---------------------------------------------------------------- preprocessing


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


def test_every_input_is_downsampled_by_block_mean(tmp_path, monkeypatch):
    """Factors of 5 and 6 on a fake release: each stored array equals the block mean of the full one."""
    cases, _, _ = load(tmp_path, monkeypatch, n=1, dataset={"nonfinite": (0,)})
    arrays, meta = cases["sub-fake0000"]
    assert meta["status"] == "ok" and meta["nonfinite_voxels"] == 6

    case = fake.add_nonfinite(fake.make_case(np.random.default_rng(0)))
    zyx = {name: data.xyz_to_zyx(a.astype(np.float32)) for name, a in case.items()}
    brain = data.brain_mask_from_ncct(zyx["ncct"])
    sl = np.where(brain.reshape(len(brain), -1).mean(1) >= CONFIG["min_brain_frac"])[0]

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
    base = np.mean(frames[: CONFIG["n_baseline"]], axis=0)
    start = meta["peak"] - CONFIG["t_pre"]
    for k, t in enumerate(np.clip(np.arange(start, start + CONFIG["t_window"]), 0, meta["T"] - 1)):
        want = blocks(np.clip(frames[t] - base, -20, 150)[sl] / 100.0)
        np.testing.assert_allclose(arrays["ctp"][k].astype(np.float32), want, atol=1e-3, err_msg=f"frame {k}")

    for name in ("ncct", "cta", "maps", "ctp"):
        assert np.isfinite(arrays[name]).all(), name


def test_every_array_goes_through_area_down(tmp_path, monkeypatch):
    calls, original = [], data.area_down

    def spy(x, size, is_mask=False):
        calls.append(is_mask)
        return original(x, size, is_mask)

    monkeypatch.setattr(data, "area_down", spy)
    load(tmp_path, monkeypatch, n=1, img_size=64)
    # 16 CTP frames, NCCT, CTA and 4 maps as images; brain and lesion as masks
    assert (calls.count(False), calls.count(True)) == (16 + 2 + 4, 2)


# ---------------------------------------------------------------- the entrypoint


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


# ---------------------------------------------------------------- gates


def smallest_effect() -> float:
    """Read from the gates and H1: the owner may change it before locking, in all places at once."""
    from labcore.config import load_direction

    document = load_direction(DIRECTION)
    text = " ".join(
        [document["gates"]["GO"], document["gates"]["PARK"], document["hypotheses"][0]["falsified_if"]]
    )
    found = re.findall(r"(?:mde149_ucl|ci_hi)\(raw-maps\) (?:<=|>=|<|>) (0\.\d+)", text)
    assert len(found) == 5 and len(set(found)) == 1, f"the gates and H1 disagree: {found}"
    return float(found[0])


def verdict_for(**in_units_of_sesoi):
    """The verdict for metrics given as multiples of the smallest effect of interest."""
    from labcore.config import load_direction
    from labcore.verdict.verdict import compute_verdict

    factors = {"mean": 1.5, "ci_hi": 3.0, "ci_lo_n1": 0.5, "mde": 0.75, **in_units_of_sesoi}
    names = {
        "mean": "mean(raw-maps)", "ci_hi": "ci_hi(raw-maps)", "ci_lo_n1": "ci_lo(raw-N1)",
        "mde": "mde149_ucl(raw-maps)",
    }
    sesoi = smallest_effect()
    values = {names[key]: factor * sesoi for key, factor in factors.items()}
    gates = load_direction(DIRECTION)["gates"]
    return compute_verdict(gates, {"pipeline_ok": True, "metrics": values})


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({}, "GO"),
        ({"mde": 1.0}, "GO"),  # exactly at the smallest effect of interest
        ({"mde": 1.05}, "PARK"),  # a 149-patient study could not detect it
        ({"ci_hi": 0.95}, "PARK"),  # the whole interval is below it
        ({"ci_hi": 1.0}, "GO"),
        ({"mean": 0.0}, "REVISE"),  # powered, not ruled out, but no positive mean
        ({"mean": -0.5}, "REVISE"),
        ({"ci_lo_n1": 0.0}, "REVISE"),  # raw does not beat the temporal-mean control
        ({"mde": 2.5, "ci_hi": 0.0}, "PARK"),
    ],
)
def test_gate_verdicts(change, expected):
    assert verdict_for(**change).verdict == expected


def test_go_and_park_are_never_both_true():
    for mean in (-0.5, 0.0, 0.5, 1.5):
        for ci_hi in (-0.5, 0.5, 0.99999, 1.0, 1.5, 3.0):
            for ci_lo_n1 in (-1.0, 0.0, 0.5):
                for mde in (0.25, 0.995, 1.0, 1.005, 1.5, 5.0):
                    result = verdict_for(mean=mean, ci_hi=ci_hi, ci_lo_n1=ci_lo_n1, mde=mde)
                    both = result.gates["GO"]["result"] and result.gates["PARK"]["result"]
                    assert not both, (mean, ci_hi, ci_lo_n1, mde)
                    assert not result.notes
