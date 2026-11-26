"""The rules, skills and docs that the lab's workflow depends on."""

import re

import pytest
import yaml
from labtestkit import REPO

SKILLS = ("new-direction", "implement", "smoke", "triage", "verdict", "graduate")


def text(path: str) -> str:
    return " ".join((REPO / path).read_text(encoding="utf-8").split())


@pytest.mark.parametrize("name", SKILLS)
def test_skill_frontmatter(name):
    body = (REPO / ".claude" / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", body, re.DOTALL)
    assert match
    front = yaml.safe_load(match.group(1))
    assert front["name"] == name and len(front["description"]) > 40


def test_claude_md_states_the_hard_rules():
    rules = text("CLAUDE.md")
    for phrase in (
        "Never commit to `main` directly", "Never edit `state/` or `results/`", "Never decide a verdict",
        "L0 test", "No secrets and no data", "FACT, INTERPRETATION, HYPOTHESIS or UNKNOWN", "800 KB",
        "Never set `gates.approval: APPROVED`",
    ):
        assert phrase in rules, phrase


def test_setup_names_every_secret_and_label():
    setup = text("SETUP.md")
    workflows = "".join(
        p.read_text(encoding="utf-8") for p in (REPO / ".github" / "workflows").glob("*.yml")
    )
    for name in set(re.findall(r"(?:secrets|vars)\.([A-Z_]+)", workflows)):
        assert name in setup, name
    for needed in ("HF_TOKEN", "run-l1", "run-failed", "UNVERIFIED", "Read and write"):
        assert needed in setup, needed


def test_relative_links_resolve():
    for document in (REPO / "README.md", REPO / "SETUP.md", REPO / "docs" / "ARCHITECTURE.md"):
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)", document.read_text(encoding="utf-8")):
            assert (document.parent / target).exists(), f"{document.name}: {target}"
