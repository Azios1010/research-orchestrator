"""new_direction, validate and verdict, plus the repo's own directions."""

import pytest
from labtestkit import REPO, TOY, approve_and_lock, copy_lab, tool

from labcore import registry
from labcore.config import ConfigError, load_direction
from labcore.io import write_text_atomic
from labcore.lab import Lab

validate = tool("validate")
new_direction = tool("new_direction")
verdict_tool = tool("verdict")


def status_of(root, slug=TOY):
    return registry.load(root / "registry.yaml")["directions"][slug]["status"]


def edit(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    write_text_atomic(path, text.replace(old, new, 1))


def test_this_repository_is_valid():
    assert validate.validate_lab(REPO) == []
    assert validate.main([]) == 0


def test_isles_gates_are_draft_and_unlocked():
    direction = load_direction(REPO / "directions" / "isles24-ctp")
    assert direction["gates"] == {
        "approval": "DRAFT",
        "GO": "mde149_ucl(raw-maps) <= 0.02 and ci_hi(raw-maps) >= 0.02 and mean(raw-maps) > 0 "
              "and ci_lo(raw-N1) > 0",
        "PARK": "mde149_ucl(raw-maps) > 0.02 or ci_hi(raw-maps) < 0.02",
    }
    assert direction["scorecard_map"]["metrics"]["mean(raw-maps)"] == {
        "file": "summary.json", "path": "contrasts.raw-maps.mean_delta",
    }
    item = registry.load(REPO / "registry.yaml")["directions"]["isles24-ctp"]
    assert item["prereg_lock"] is None and item["status"] in ("SCAFFOLDED", "L0_PASS")
    assert direction["data"]["contains_patient_data"] is True


def test_new_direction(lab_root):
    target = new_direction.new_direction(lab_root, TOY, 'Does "A" beat B: yes or no?')
    assert validate.validate_lab(lab_root) == []
    assert status_of(lab_root) == "SCAFFOLDED"
    assert load_direction(target)["title"] == 'Does "A" beat B: yes or no?'
    assert load_direction(target)["gates"]["approval"] == "DRAFT"
    for path in target.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".txt"}:
            assert "__SLUG__" not in path.read_text(encoding="utf-8"), path


@pytest.mark.parametrize("slug", ["Bad", "has_underscore", "-x", "x-", "a" * 31, "../up", ""])
def test_bad_slugs(lab_root, slug):
    with pytest.raises(new_direction.ScaffoldError):
        new_direction.new_direction(lab_root, slug, "t")


def test_an_existing_direction_is_not_overwritten(toy_lab):
    with pytest.raises(new_direction.ScaffoldError):
        new_direction.new_direction(toy_lab, TOY, "again")
    assert new_direction.main([TOY, "--title", "x", "--root", str(toy_lab)]) == 1


@pytest.mark.parametrize(
    "file, old, new, message",
    [
        ("direction.yaml", "status: SCAFFOLDED", "status: L0_PASS", "differs from registry"),
        ("direction.yaml", f"slug: {TOY}", "slug: other-name", "slug is"),
        ("direction.yaml", 'GO: "ci_lo(a-b) > 0"', 'GO: "ci_lo(x-y) > 0"', "ci_lo(x-y)"),
        ("direction.yaml", 'GO: "ci_lo(a-b) > 0"', 'GO: "__import__(os) > 0"', "gate"),
        ("direction.yaml", "approval: DRAFT", "approval: MAYBE", "approval"),
        ("configs/exp_001.yaml", "est_gpu_hours: 0.5", "est_gpu_hours: 500", "exceeds"),
    ],
)
def test_validation_errors(toy_lab, file, old, new, message):
    edit(toy_lab / "directions" / TOY / file, old, new)
    errors = "\n".join(validate.validate_lab(toy_lab))
    assert message.lower() in errors.lower(), errors
    assert validate.main(["--root", str(toy_lab)]) == 1


def test_an_unregistered_or_incomplete_folder(toy_lab):
    (toy_lab / "directions" / "stray").mkdir()
    (toy_lab / "directions" / TOY / "checks.py").unlink()
    errors = "\n".join(validate.validate_lab(toy_lab))
    assert "stray is not registered" in errors and "missing checks.py" in errors
    assert "stray" not in "\n".join(validate.validate_lab(toy_lab, [TOY]))


def test_unknown_direction_fields_are_rejected(toy_lab):
    directory = toy_lab / "directions" / TOY
    with open(directory / "direction.yaml", "a", encoding="utf-8") as handle:
        handle.write("surprise: 1\n")
    with pytest.raises(ConfigError, match="surprise"):
        load_direction(directory)


def test_l0_pass_is_recorded_and_withdrawn(toy_lab):
    passed, message = validate.run_l0(toy_lab, TOY)
    assert passed, message
    assert status_of(toy_lab) == "L0_PASS"
    record = registry.load(toy_lab / "registry.yaml")["directions"][TOY]["l0"]
    assert set(record) == {"passed_at", "code_hash", "labcore_hash"}
    assert tool("lock_prereg").main([TOY, "--root", str(toy_lab)]) == 1  # gates are still DRAFT

    test_file = toy_lab / "directions" / TOY / "tests" / "test_l0_smoke.py"
    with open(test_file, "a", encoding="utf-8") as handle:
        handle.write("\n\ndef test_broken():\n    assert False\n")
    passed, _ = validate.run_l0(toy_lab, TOY)
    assert not passed and status_of(toy_lab) == "SCAFFOLDED"


def test_skipped_l0_tests_are_not_a_pass(toy_lab):
    (toy_lab / "directions" / TOY / "tests" / "test_l0_smoke.py").write_text(
        "import pytest\n\ndef test_l0_smoke():\n    pytest.importorskip('surely_not_installed')\n"
    )
    passed, message = validate.run_l0(toy_lab, TOY)
    assert not passed and "skipped" in message


def test_no_record_leaves_the_registry_alone(toy_lab):
    before = (toy_lab / "registry.yaml").read_bytes()
    assert validate.main(["--root", str(toy_lab), "--run-l0", TOY, "--no-record"]) == 0
    assert (toy_lab / "registry.yaml").read_bytes() == before


def test_verdict_tool_refuses_without_a_run_or_a_lock(toy_lab):
    with pytest.raises(verdict_tool.VerdictError, match="SCAFFOLDED"):
        verdict_tool.record_verdict(Lab.at(toy_lab), TOY)
    approve_and_lock(toy_lab, TOY)
    with pytest.raises(verdict_tool.VerdictError, match="no usable run"):
        verdict_tool.record_verdict(Lab.at(toy_lab), TOY)
    edit(toy_lab / "directions" / TOY / "direction.yaml", 'GO: "ci_lo(a-b) > 0"', 'GO: "ci_lo(a-b) > -9"')
    with pytest.raises(verdict_tool.VerdictError, match="differs from the lock"):
        verdict_tool.record_verdict(Lab.at(toy_lab), TOY)
    assert verdict_tool.main([TOY, "--root", str(toy_lab)]) == 1


def test_copy_lab_helper_copies_real_directions(tmp_path):
    root = copy_lab(tmp_path / "copy", ("isles24-ctp",))
    assert validate.validate_lab(root) == []
