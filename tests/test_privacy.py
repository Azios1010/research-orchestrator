"""No phenotype columns in results/: the name list, the file check and every place that applies it."""

import json

import pytest
from labtestkit import (
    REPO,
    TOY,
    RecordingNotifier,
    approve_and_lock,
    make_watcher,
    replace_pipeline,
    runs,
    tool,
)

from labcore.config import ConfigError, load_direction
from labcore.io import utcnow, write_text_atomic
from labcore.kaggle.local_backend import LocalBackend
from labcore.lab import Lab
from labcore.privacy import file_problems, is_phenotype_name, names_in, tree_problems
from labcore.runs.dispatch import dispatch
from labcore.watcher.checks import copy_results, generic_checks

privacy_check = tool("results_privacy")


@pytest.mark.parametrize(
    "name",
    [
        "age", "Age", "AGE", "patient_age", "PatientAge", "patientage", "AgeAtScan", "age_years",
        "agegroup", "age (years)", "sex", "Sex", "patient_sex", "gender", "NIHSS", "nihss_admission",
        "baselineNIHSS", "nihss24h", "mrs", "mRS", "mrs_90d", "mRS90", "premorbid_mRS", "phenotype",
        "phenotype_cluster", "onset_to_door_min", "time_since_onset", "atrial_fibrillation",
        "hypertension", "diabetes", "smoking_status", "BMI", "date_of_birth", "HbA1c", "patient_name",
        "etiology", "mTICI", "ethnicity", "weight_kg", "mean(age-maps)",
    ],
)
def test_phenotype_names(name):
    assert is_phenotype_name(name)


@pytest.mark.parametrize(
    "name",
    [
        # columns and keys that runs really write
        "subject_id", "arm", "dice", "gt_ml", "pred_ml", "abs_vol_err_ml", "fold", "status", "T", "peak",
        "n_slices", "gt_ml_full", "vox_ml_down", "shape", "shape_ctp", "zooms", "shard", "nonfinite_ctp",
        "nonfinite_voxels", "contrast", "n", "mean_delta", "ci95_lo", "sd_delta", "share_patients_a_better",
        "MDE_at_n149 (2.8*sd/sqrt149)", "lesion_size", "maps", "raw", "N1", "N2", "mean(raw-maps)",
        "ci_lo(raw-N1)", "mde149(raw-maps)", "stage", "ok", "error_message", "traceback_tail", "gpu_name",
        "torch_version", "labcore_version", "n_boot", "lr", "img_size", "min_brain_frac", "repo_id",
        # words that merely contain a listed one
        "image_size", "usage", "average", "coverage", "voltage", "percentage", "storage_mb", "image_group",
        "stage_group", "message", "page", "smoke_test", "class_weight", "image_height", "manager",
    ],
)
def test_ordinary_names(name):
    assert not is_phenotype_name(name)


@pytest.mark.parametrize(
    "filename, content",
    [
        ("a.csv", "subject_id,age,dice\nsub-1,67,0.5\n"),
        ("a.csv", "\ufeffSex,dice\nF,0.5\n"),
        ("a.csv", "\n\nsubject_id;NIHSS\nsub-1;67\n"),
        ("a.csv", '"subject id","mRS 90"\n"sub-1","67"\n'),
        ("a.tsv", "subject_id\tgender\nsub-1\t67\n"),
        ("a.json", '{"strata": [{"sex": "F", "n": 67}]}'),
        ("a.json", '{"columns": ["subject_id", "NIHSS"], "data": [["sub-1", 67]]}'),
        ("a.json", '{"schema": {"fields": [{"name": "age"}]}, "data": []}'),
        ("a.jsonl", '{"dice": 1}\n\n{"age": 67}\n'),
        ("A.CSV", "Age\n67\n"),
    ],
)
def test_files_with_phenotype_columns_fail_without_quoting_values(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    [problem] = file_problems(path)
    assert "phenotype-like column" in problem
    assert "67" not in problem and "sub-1" not in problem


@pytest.mark.parametrize(
    "filename, content",
    [("a.json", "{not json"), ("a.jsonl", "{}\nnope\n"), ("a.csv", b"\xff\xfe\x00a")],
)
def test_a_file_that_cannot_be_read_fails(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    assert "cannot be read" in file_problems(path)[0]


def test_clean_and_unchecked_files_pass(tmp_path):
    (tmp_path / "per_patient.csv").write_text("subject_id,dice\nage,0.5\n")  # a value, not a column
    (tmp_path / "empty.csv").write_text("")
    (tmp_path / "s.json").write_text('{"note": "age", "values": [1, 2]}')
    (tmp_path / "error_log.txt").write_text("age,sex\n")
    assert tree_problems(tmp_path) == []
    assert names_in({"metrics": {"mean(raw-maps)": 1}}) == []


def test_ci_tool(tmp_path, capsys):
    assert privacy_check.main(["--root", str(tmp_path)]) == 0  # no results/ folder yet
    folder = tmp_path / "results" / "d" / "e" / "r"
    folder.mkdir(parents=True)
    (folder / "per_patient.csv").write_text("subject_id,dice\ns,1\n")
    assert privacy_check.main(["--root", str(tmp_path)]) == 0
    (folder / "cases.csv").write_text("subject_id,nihss\ns,4\n")
    assert privacy_check.main(["--root", str(tmp_path)]) == 1
    assert "results/d/e/r/cases.csv" in capsys.readouterr().err


def test_this_repository_passes():
    assert privacy_check.main(["--root", str(REPO)]) == 0


def test_the_watcher_store_refuses_the_file(tmp_path):
    out, result = tmp_path / "out", tmp_path / "result"
    out.mkdir()
    (out / "status.json").write_text('{"ok": true}')
    (out / "bad.csv").write_text("subject_id,age\ns,67\n")
    (out / "good.csv").write_text("subject_id,dice\ns,1\n")
    problems = copy_results(out, result, ["bad.csv", "good.csv"])
    assert len(problems) == 1 and "not stored: bad.csv" in problems[0]
    assert sorted(p.name for p in result.iterdir()) == ["good.csv", "status.json"]


def test_generic_checks_flag_the_file_so_l0_fails(tmp_path):
    (tmp_path / "status.json").write_text('{"ok": true}')
    (tmp_path / "bad.csv").write_text("subject_id,age\ns,67\n")
    [problem] = generic_checks(tmp_path, ["bad.csv"])
    assert "'age'" in problem


def run_and_watch(root, tmp_path):
    approve_and_lock(root, TOY)
    backend = LocalBackend(tmp_path / "kernels")
    dispatch(Lab.at(root), TOY, "exp_001", backend, require_lock=True)
    notifier = RecordingNotifier()
    watcher = make_watcher(root, {"local": backend}, notifier, clock=utcnow)
    watcher.run_once()
    assert watcher.errors == []
    [run] = runs(root)
    return run, root / run["result_path"], notifier


def test_a_run_writing_phenotypes_is_invalid_and_nothing_of_it_is_stored(toy_lab, tmp_path):
    summary = {"n_requested": 8, "n_usable": 8,
               "contrasts": {"a-b": {"mean_delta": 0.1, "ci_lo": 0.05, "ci_hi": 0.2}}}
    replace_pipeline(toy_lab, f"""
from pathlib import Path
out = Path(out_dir)
(out / "per_subject.csv").write_text("subject_id,a,b,age\\nsubject_000,1,0,6789\\n")
(out / "summary.json").write_text({json.dumps(summary)!r})
""")
    run, folder, notifier = run_and_watch(toy_lab, tmp_path)

    assert run["status"] == "COMPLETE_INVALID" and run["retries"] == 0
    assert not (folder / "per_subject.csv").exists()
    assert not (folder / "scorecard.json").exists() and not (folder / "verdict.json").exists()
    stored = "".join(p.read_text(encoding="utf-8") for p in folder.rglob("*") if p.is_file())
    assert "phenotype-like column" in stored and "6789" not in stored
    assert "6789" not in json.dumps(list(notifier.issues.values()))
    assert tree_problems(toy_lab / "results") == []


def test_a_phenotype_metric_name_is_rejected_before_any_run(toy_lab):
    path = toy_lab / "directions" / TOY / "direction.yaml"
    text = path.read_text(encoding="utf-8")
    metric = '    "mean_age": {file: summary.json, path: x}'
    marker = '    "ci_hi(a-b)":'
    assert marker in text
    write_text_atomic(path, text.replace(marker, "\n".join((metric, marker))))
    with pytest.raises(ConfigError, match="phenotype"):
        load_direction(toy_lab / "directions" / TOY)


def test_results_columns_refuse_a_neutral_name_that_the_blocklist_cannot_know(tmp_path):
    table = tmp_path / "per_subject.csv"
    table.write_text("subject_id,arm,dice,var_7\n1,raw,0.5,63\n", encoding="utf-8")
    assert file_problems(table, "per_subject.csv") == []  # nothing in the name list matches var_7
    allowed = {"per_subject.csv": ["subject_id", "arm", "dice"]}
    [problem] = file_problems(table, "per_subject.csv", allowed)
    assert "var_7" in problem and "results_columns" in problem and "63" not in problem
    table.write_text("subject_id,arm,dice\n1,raw,0.5\n", encoding="utf-8")
    assert file_problems(table, "per_subject.csv", allowed) == []


def test_results_columns_need_every_table_output_listed(tmp_path):
    table = tmp_path / "other.csv"
    table.write_text("a,b\n1,2\n", encoding="utf-8")
    [problem] = file_problems(table, "other.csv", {"per_subject.csv": ["a", "b"]})
    assert "does not list" in problem
    summary = tmp_path / "summary.json"
    summary.write_text('{"n": 1}', encoding="utf-8")  # JSON is not covered by the table allowlist
    assert file_problems(summary, "summary.json", {"per_subject.csv": []}) == []


def test_tree_problems_and_the_ci_script_apply_results_columns(tmp_path):
    results = tmp_path / "results" / "toy" / "exp_001" / "run-1"
    results.mkdir(parents=True)
    (results / "per_subject.csv").write_text("subject_id,var_7\n1,63\n", encoding="utf-8")
    assert tree_problems(tmp_path / "results") == []
    assert tree_problems(results, {"per_subject.csv": ["subject_id"]})
    directions = tmp_path / "directions"
    directions.mkdir()
    import shutil
    shutil.copytree(REPO / "directions" / "_template", directions / "toy")
    text = (directions / "toy" / "direction.yaml").read_text(encoding="utf-8")
    text = text.replace("__SLUG__", "toy").replace("__TITLE_JSON__", '"Toy"')
    (directions / "toy" / "direction.yaml").write_text(
        text + "results_columns:\n  per_subject.csv: [subject_id]\n", encoding="utf-8"
    )
    found = privacy_check.results_problems(tmp_path)
    assert any("var_7" in p and "results_columns" in p for p in found)
    (results / "per_subject.csv").write_text("subject_id\n1\n", encoding="utf-8")
    assert privacy_check.results_problems(tmp_path) == []


def test_the_direction_schema_accepts_results_columns():
    import jsonschema

    from labcore.schema import load_schema
    schema = load_schema("direction")
    props = schema["properties"]
    assert "results_columns" in props
    jsonschema.validate({"per_subject.csv": ["subject_id", "dice"]}, props["results_columns"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"per_subject.csv": "subject_id"}, props["results_columns"])


def test_the_watcher_refuses_to_store_an_unlisted_column(tmp_path):
    out, stored = tmp_path / "out", tmp_path / "stored"
    out.mkdir()
    (out / "status.json").write_text('{"ok": true}', encoding="utf-8")
    (out / "per_subject.csv").write_text("subject_id,var_7\n1,63\n", encoding="utf-8")
    columns = {"per_subject.csv": ["subject_id", "dice"]}
    problems = copy_results(out, stored, ["per_subject.csv"], columns)
    assert any("var_7" in p and "not stored" in p for p in problems)
    assert not (stored / "per_subject.csv").exists()
    assert (stored / "status.json").exists()
    assert copy_results(out, tmp_path / "plain", ["per_subject.csv"]) == []  # no allowlist, old behaviour
