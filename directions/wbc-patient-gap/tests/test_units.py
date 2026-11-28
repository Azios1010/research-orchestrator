"""Unit tests: checkpoint loading, data rules, splits, statistics, gates. No network."""

import importlib
import itertools
import zipfile
from pathlib import Path

import numpy as np
import pytest

for module in ("torch", "sklearn", "PIL"):
    pytest.importorskip(module)

import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from labcore.config import load_direction  # noqa: E402
from labcore.testing import load_module, load_package  # noqa: E402
from labcore.verdict.verdict import compute_verdict  # noqa: E402

DIRECTION = Path(__file__).resolve().parents[1]
package = load_package(DIRECTION / "src")
backbone, data, evaluate, main, probe, splits = (
    importlib.import_module(f"{package}.{name}")
    for name in ("backbone", "data", "evaluate", "main", "probe", "splits")
)
fake = load_module(DIRECTION / "tests" / "fake_data.py")


# ---------------------------------------------------------------- checkpoint


def teacher_checkpoint():
    return {
        "teacher": {
            "backbone.cls_token": torch.ones(1, 1, 384),
            "backbone.pos_embed": torch.full((1, 257, 384), 2.0),
            "backbone.blocks.0.weight": torch.full((3,), 3.0),
            "dino_head.mlp.weight": torch.zeros(2),
            "ibot_head.mlp.weight": torch.zeros(2),
        }
    }


def test_keys_lose_the_prefix_and_the_heads():
    state = backbone.convert_teacher_state_dict(teacher_checkpoint())
    assert sorted(state) == ["blocks.0.weight", "cls_token", "pos_embed"]


@pytest.mark.parametrize(
    "checkpoint, message",
    [
        ({"student": {}}, "no 'teacher' entry"),
        ({"teacher": {"dino_head.w": 1}}, "no backbone weight"),
        ({"teacher": {"backbone.w": 1, "w": 2}}, "two checkpoint keys"),
    ],
)
def test_unexpected_checkpoints_are_refused(checkpoint, message):
    with pytest.raises(ValueError, match=message):
        backbone.convert_teacher_state_dict(checkpoint)


class Hub(torch.nn.Module):
    """The names and shapes that matter: a 518 pixel position embedding, as the hub model has."""

    def __init__(self):
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 1370, 384))
        self.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.blocks[0].weight = torch.nn.Parameter(torch.zeros(3))


def test_the_position_embedding_is_replaced_and_every_weight_loads():
    model = backbone.load_teacher(Hub(), teacher_checkpoint())
    assert tuple(model.pos_embed.shape) == (1, 257, 384)
    assert float(model.pos_embed.detach().mean()) == 2.0
    assert float(model.blocks[0].weight.detach().sum()) == 9.0


@pytest.mark.parametrize("drop", ["backbone.cls_token", "backbone.blocks.0.weight"])
def test_loading_is_strict(drop):
    checkpoint = teacher_checkpoint()
    del checkpoint["teacher"][drop]
    with pytest.raises(RuntimeError, match="Missing key"):
        backbone.load_teacher(Hub(), checkpoint)
    checkpoint = teacher_checkpoint()
    checkpoint["teacher"]["backbone.extra"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        backbone.load_teacher(Hub(), checkpoint)


def test_preprocessing_resizes_and_normalises(tmp_path):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (360, 363), 255).save(buffer, format="JPEG")  # grey input becomes RGB
    array = backbone.preprocess(buffer.getvalue())
    assert array.shape == (3, 224, 224) and array.dtype == np.float32
    expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(array, np.broadcast_to(expected[:, None, None], array.shape), rtol=1e-6)


# ---------------------------------------------------------------- data rules


def test_pinned_sources():
    for source in (data.DEEPAPL, data.ACEVEDO, backbone.WEIGHTS):
        assert source.url.startswith("https://") and len(source.sha256) == 64 and source.size > 1e8
        int(source.sha256, 16)
    assert len(backbone.DINOV2_COMMIT) == 40


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Patient_03/Signed slides/Myelocyte/a.jpg", ("Patient_03", "Myelocyte")),
        ("Patient_03/123/Signed slides/Myelocyte/a.JPG", ("Patient_03", "Myelocyte")),
        ("Data/All/Patient_03/x/Signed slides/Myelocyte/a.jpg", ("Patient_03", "Myelocyte")),
        ("Patient_03/Unsigned slides/a.jpg", None),
        ("Patient_03/Signed slides/Myelocyte/Thumbs.db", None),
        ("Patient_03/Signed slides/Myelocyte/._a.jpg", None),
        ("__MACOSX/Patient_03/Signed slides/Myelocyte/a.jpg", None),
        ("Patient_03/Signed slides/a.jpg", None),
        ("Patient_03/Signed slides/Myelocyte/deeper/a.jpg", None),
    ],
)
def test_which_entries_are_signed_images(name, expected):
    assert data._signed_entry(name) == expected


@pytest.mark.parametrize(
    "name", ["x/Signed slides/Myelocyte/a.jpg", "Patient_1/Patient_2/Signed slides/Myelocyte/a.jpg"]
)
def test_a_signed_image_needs_exactly_one_patient_folder(name):
    with pytest.raises(data.DataError) as error:
        data._signed_entry(name)
    assert "Patient_" not in str(error.value)


def test_duplicate_rules():
    records = [
        ("b", "p1", "Myelocyte", "h1"),  # repeated within a class: the first by name stays
        ("a", "p2", "Myelocyte", "h1"),
        ("c", "p1", "Myelocyte", "h2"),  # under two classes: both go
        ("d", "p1", "Basophil", "h2"),
        ("e", "p1", "Basophil", "h3"),
    ]
    kept, stats = data.deduplicate(records)
    assert [record[0] for record in kept] == ["a", "e"]
    assert stats == {
        "conflicting_groups": 1, "conflicting_images": 2, "repeated_images_dropped": 1,
        "repeats_across_patients": 1,
    }


def test_the_fake_archive_gives_the_expected_counts(tmp_path):
    made = fake.make_dataset(tmp_path, n_patients=8, cells_per_class=3)
    assert fake.CLASSES == data.CLASSES
    assert set(fake.ACEVEDO) == set(data.ACEVEDO_FOLDERS)
    cells = data.index_deepapl(made["deepapl"])
    assert data.class_counts(cells) == made["expected_deepapl"]
    assert cells.stats["conflicting_groups"] == 2
    assert cells.stats["repeated_images_dropped"] == 1
    assert len(set(cells.hashes.tolist())) == len(cells.names)
    assert not any("Unsigned" in name or fake.OTHER_CLASS in name for name in cells.names)
    names, folders = data.index_acevedo(made["acevedo"])
    assert {f: folders.count(f) for f in folders} == made["expected_acevedo"]
    assert not any("__MACOSX" in name or "/." in name for name in names)


def test_errors_from_the_archive_do_not_name_files(tmp_path):
    path = tmp_path / "broken.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Patient_07/Signed slides/Myelocyte/secret_name.jpg", b"x" * 64)
    raw = bytearray(path.read_bytes())
    raw[raw.index(b"x" * 64)] = ord("y")  # the stored bytes no longer match their CRC
    path.write_bytes(bytes(raw))
    with pytest.raises(data.DataError) as error:
        data.index_deepapl(path)
    assert "Patient_" not in str(error.value) and "secret_name" not in str(error.value)
    assert error.value.__suppress_context__


def test_an_undecodable_image_is_reported_without_its_name(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Patient_07/secret_name.jpg", b"not an image")
    with pytest.raises(data.DataError) as error:
        backbone.extract(
            backbone.Tiny(), path, ["Patient_07/secret_name.jpg"], batch_size=4, workers=1, device="cpu",
            dataset="DeepAPL",
        )
    assert "secret_name" not in str(error.value) and error.value.__suppress_context__


def source_for(path: Path, sha256: str | None = None) -> "data.Source":
    return data.Source("copy.bin", path.as_uri(), path.stat().st_size, sha256 or data.sha256_file(path))


def test_download_checks_what_it_got(tmp_path):
    original = tmp_path / "original.bin"
    original.write_bytes(b"content")
    target = data.download(source_for(original), tmp_path / "cache", wait_s=0)
    assert target.read_bytes() == b"content"
    assert data.download(source_for(original), tmp_path / "cache", wait_s=0) == target
    with pytest.raises(data.DataError, match="download failed after 2 attempts.*sha256"):
        data.download(source_for(original, "0" * 64), tmp_path / "other", attempts=2, wait_s=0)
    assert not list((tmp_path / "other").iterdir())


def test_download_refuses_a_web_page(tmp_path):
    page = tmp_path / "challenge.html"
    page.write_text("<html>are you human?</html>")
    with pytest.raises(data.DataError, match="instead of the file"):
        data.download(source_for(page), tmp_path / "cache", attempts=1, wait_s=0)


# ---------------------------------------------------------------- parameters


def write_config(tmp_path, **params):
    base = {"expected_deepapl": {"a": [1, 1]}, "expected_acevedo": {"ig": 1}}
    import yaml

    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"params": {**base, **params}}))
    return str(path)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"surprise": 1}, "unknown config parameters"),
        ({"backbone": "tiny"}, "offline testing"),
        ({"deepapl_zip": "a.zip"}, "both or neither"),
        ({"split_seeds": [1, 1]}, "distinct"),
        ({"expected_deepapl": {}}, "expected counts"),
        ({"backbone": "other"}, "backbone must be"),
    ],
)
def test_bad_parameters_are_refused(tmp_path, params, message):
    with pytest.raises(ValueError, match=message):
        main.load_params(write_config(tmp_path, **params))


def test_the_real_config_has_the_pre_registered_settings():
    cfg = main.load_params(str(DIRECTION / "configs" / "exp_001_pilot.yaml"))
    assert cfg["backbone"] == "dinobloom-s" and not cfg["deepapl_zip"] and not cfg["acevedo_zip"]
    assert (cfg["n_folds"], len(cfg["split_seeds"]), cfg["n_boot"], cfg["knn_k"]) == (5, 3, 10000, 20)
    assert sum(images for images, _ in cfg["expected_deepapl"].values()) == 12364
    assert tuple(cfg["expected_deepapl"]) == data.CLASSES
    assert sum(cfg["expected_acevedo"].values()) == 17092


# ---------------------------------------------------------------- splits


def cohort(seed=0, patients=12):
    rng = np.random.default_rng(seed)
    who = np.repeat([f"p{i:02d}" for i in range(patients)], rng.integers(5, 40, patients))
    return who, rng.integers(0, 4, len(who))


def test_folds_are_by_patient_and_balanced():
    who, _ = cohort()
    folds = splits.patient_folds(who, 5, 7)
    assert sorted(folds) == sorted(set(who.tolist()))
    sizes = [list(folds.values()).count(f) for f in range(5)]
    assert max(sizes) - min(sizes) <= 1 and min(sizes) > 0
    assert folds == splits.patient_folds(who, 5, 7) and folds != splits.patient_folds(who, 5, 8)
    with pytest.raises(ValueError, match="cannot fill"):
        splits.patient_folds(who[who == "p00"], 5, 7)


def test_halves_are_disjoint_complete_and_stratified():
    who, labels = cohort()
    cells = np.flatnonzero(np.isin(who, ["p00", "p03", "p07"]))
    evaluated, donors = splits.split_eval_donor(cells, who, labels, np.random.default_rng(1))
    assert not np.intersect1d(evaluated, donors).size
    np.testing.assert_array_equal(np.union1d(evaluated, donors), cells)
    for patient in ("p00", "p03", "p07"):
        a, b = evaluated[who[evaluated] == patient], donors[who[donors] == patient]
        assert abs(len(a) - len(b)) <= 1
        for k in range(4):
            assert abs((labels[a] == k).sum() - (labels[b] == k).sum()) <= 1
    again = splits.split_eval_donor(cells, who, labels, np.random.default_rng(1))
    np.testing.assert_array_equal(again[0], evaluated)


def test_a_single_cell_goes_to_one_half():
    evaluated, donors = splits.split_eval_donor(
        np.array([0]), np.array(["p"]), np.array([2]), np.random.default_rng(0)
    )
    assert len(evaluated) + len(donors) == 1


def test_shared_train_swaps_cells_class_by_class():
    labels = np.array([0] * 10 + [1] * 2 + [2] * 5 + [0] * 3 + [1] * 6)
    disjoint, donors = np.arange(17), np.arange(17, 26)
    shared = splits.shared_train(disjoint, donors, labels, np.random.default_rng(0))
    assert len(shared) == len(disjoint) == len(np.unique(shared))
    np.testing.assert_array_equal(np.bincount(labels[shared]), np.bincount(labels[disjoint]))
    swapped = np.intersect1d(shared, donors)
    assert (labels[swapped] == 0).sum() == 3  # all donors of a large class
    assert (labels[swapped] == 1).sum() == 2  # only as many as the class has in the train set
    assert (labels[swapped] == 2).sum() == 0  # no donor: nothing changes


# ---------------------------------------------------------------- statistics


def test_f1_from_patient_counts_matches_sklearn():
    rng = np.random.default_rng(3)
    unit, true, pred = rng.integers(0, 9, 500), rng.integers(0, 12, 500), rng.integers(0, 12, 500)
    table = evaluate.counts_by_unit(unit, true, pred, 9, 12)
    assert table[..., 0].sum() + table[..., 2].sum() == 500
    expected = f1_score(true, pred, labels=list(range(12)), average=None, zero_division=0)
    np.testing.assert_allclose(evaluate.f1_from_counts(table.sum(axis=0)), expected)
    classes = evaluate.PRIMARY_INDEX
    assert evaluate.macro_f1(table.sum(axis=0), classes) == pytest.approx(expected[list(classes)].mean())


def test_primary_classes():
    assert [data.CLASSES[i] for i in evaluate.PRIMARY_INDEX] == [
        "Segmented neutrophils", "Band neutrophils", "Metamyelocyte", "Myelocyte", "Promyelocyte"
    ]


def tables(seed=0, patients=30, better=0.0):
    """(seeds, patients, 12, 3) tables; `better` is the extra share of right answers."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        unit = rng.integers(0, patients, 1500)
        true = rng.integers(0, 12, 1500)
        wrong = rng.integers(0, 12, 1500)
        pred = np.where(rng.random(1500) < 0.5 + better, true, wrong)
        out.append(evaluate.counts_by_unit(unit, true, pred, patients, 12))
    return np.stack(out)


def test_bootstrap_of_identical_arms_is_zero():
    a = tables()
    weights = evaluate.bootstrap_weights(30, 200, np.random.default_rng(1))
    assert (weights.sum(axis=1) == 30).all()
    out = evaluate.paired_bootstrap(a, a, evaluate.PRIMARY_INDEX, weights)
    assert out["mean_delta"] == out["ci_lo"] == out["ci_hi"] == 0.0


def test_bootstrap_point_interval_and_sign():
    a, b = tables(better=0.0), tables(better=0.25)
    weights = evaluate.bootstrap_weights(30, 2000, np.random.default_rng(1))
    out = evaluate.paired_bootstrap(a, b, evaluate.PRIMARY_INDEX, weights)
    expected = evaluate.arm_f1(b, evaluate.PRIMARY_INDEX) - evaluate.arm_f1(a, evaluate.PRIMARY_INDEX)
    assert out["mean_delta"] == pytest.approx(expected) and len(out["by_seed"]) == 3
    assert 0 < out["ci_lo"] <= out["mean_delta"] <= out["ci_hi"]
    # One resample by hand: patients repeated according to the weights.
    drawn = np.repeat(np.arange(30), weights[0])
    manual = evaluate.arm_f1(b[:, drawn], evaluate.PRIMARY_INDEX)
    summed = np.tensordot(weights[:1], b, axes=(1, 1))
    assert evaluate.macro_f1(summed, evaluate.PRIMARY_INDEX).mean() == pytest.approx(manual)
    with pytest.raises(ValueError, match="equal"):
        evaluate.paired_bootstrap(a, b[:, :10], evaluate.PRIMARY_INDEX, weights)


def test_seven_groups():
    labels = np.arange(12)
    groups = dict(zip(data.CLASSES, evaluate.group_of_label(labels).tolist(), strict=True))
    assert groups["Blast, no lineage spec"] == -1
    assert groups["Segmented neutrophils"] == groups["Band neutrophils"] == 0
    assert groups["Lymphocyte"] == groups["Lymphocyte, variant"]
    assert groups["Promyelocyte"] == groups["Myelocyte"] == groups["Metamyelocyte"]
    assert sorted(set(groups.values())) == [-1, *range(7)]
    probabilities = np.zeros((1, 12))
    probabilities[0, data.CLASSES.index("Blast, no lineage spec")] = 0.5  # in no group
    probabilities[0, data.CLASSES.index("Band neutrophils")] = 0.15
    probabilities[0, data.CLASSES.index("Segmented neutrophils")] = 0.15
    probabilities[0, data.CLASSES.index("Monocyte")] = 0.2
    assert evaluate.GROUP_NAMES[evaluate.group_prediction(probabilities)[0]] == "neutrophil"


def test_neighbour_shares_by_hand():
    train_patients = np.array(["a", "a", "b", "b"])
    train_labels = np.array([0, 1, 0, 0])
    neighbours = np.array([[0, 1]])  # both from patient a
    out = evaluate.neighbour_shares(neighbours, train_patients, train_labels, np.array(["a"]))
    # observed 1; at random 2/4; same class: (1/3 for class 0 + 1/1 for class 1) / 2
    np.testing.assert_allclose(out, [1.0, 0.5, (1 / 3 + 1) / 2, 1])


def test_knn_votes_and_breaks_ties_by_similarity():
    train = probe.l2_normalise(np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.6, 0.8]]))
    labels = np.array([0, 0, 1, 1])
    query = probe.l2_normalise(np.array([[1.0, 0.05], [0.0, 1.0], [1.0, 1.0]]))
    predicted, neighbours = probe.knn(train, labels, query, 3, 3)
    assert predicted[:2].tolist() == [0, 1] and neighbours.shape == (3, 3)
    two, _ = probe.knn(train, labels, query[2:], 4, 3)  # two votes each
    assert two.tolist() == [1]
    with pytest.raises(ValueError, match="at least 5"):
        probe.knn(train, labels, query, 5, 3)


def test_probe_gives_absent_classes_no_probability():
    rng = np.random.default_rng(0)
    x = probe.l2_normalise(rng.normal(size=(90, 8)) + np.repeat(np.eye(8)[:3] * 4, 30, axis=0))
    y = np.repeat([0, 2, 5], 30)
    (out,), iterations, converged = probe.fit_predict(
        x, y, [x], 12, c=1.0, class_weight="balanced", max_iter=500, tol=1e-4
    )
    assert converged and iterations > 0 and out.shape == (90, 12)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert (out[:, [1, 3, 4, 6]] == 0).all() and (out.argmax(axis=1) == y).mean() > 0.9
    _, _, stopped = probe.fit_predict(
        x, y, [x], 12, c=100.0, class_weight=None, max_iter=1, tol=1e-12
    )
    assert stopped is False


# ---------------------------------------------------------------- gates


def verdict_for(mean, ci_lo, ci_hi):
    gates = load_direction(DIRECTION)["gates"]
    scorecard = {
        "n_subjects_requested": 106, "n_subjects_usable": 106, "pipeline_ok": True, "gpu_hours": 0.1,
        "runtime_min": 1.0,
        "metrics": {
            "mean(shared-disjoint)": mean, "ci_lo(shared-disjoint)": ci_lo, "ci_hi(shared-disjoint)": ci_hi,
        },
    }
    return compute_verdict(gates, scorecard)


GRID = (-0.5, -0.02, -1e-9, 0.0, 1e-9, 0.0199, 0.02, 0.0201, 0.05, 0.5)


def test_go_and_park_are_never_both_true_for_an_interval_that_holds_its_estimate():
    checked = 0
    for ci_lo, mean, ci_hi in itertools.combinations_with_replacement(GRID, 3):
        result = verdict_for(mean, ci_lo, ci_hi)
        assert not (result.gates["GO"]["result"] and result.gates["PARK"]["result"]), (ci_lo, mean, ci_hi)
        assert not result.notes
        checked += 1
    assert checked == 220


def test_an_estimate_above_its_own_interval_ends_as_revise():
    """Possible in principle for a percentile bootstrap: the lab's verdict code then says REVISE."""
    result = verdict_for(mean=0.03, ci_lo=0.001, ci_hi=0.01)
    assert result.verdict == "REVISE" and "conflict" in result.notes[0]


@pytest.mark.parametrize(
    "values, verdict",
    [
        ((0.03, 0.01, 0.05), "GO"),
        ((0.02, 0.001, 0.04), "GO"),
        ((0.015, 0.001, 0.03), "REVISE"),
        ((0.03, -0.001, 0.06), "REVISE"),
        ((0.005, -0.01, 0.019), "PARK"),
        ((0.01, 0.005, 0.015), "PARK"),
        ((0.0, -0.02, 0.02), "REVISE"),
    ],
)
def test_gate_examples(values, verdict):
    assert verdict_for(*values).verdict == verdict
