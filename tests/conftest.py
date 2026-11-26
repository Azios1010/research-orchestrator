"""Shared fixtures. No test may reach GitHub, Kaggle, DeepSeek or Telegram, or touch this repo's
git history: credentials are removed, git and gh get private config folders, and every lab or git
repository used by a test is a throwaway copy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from labtestkit import SECRET_PREFIXES, TOY, copy_lab, tool

from labcore.modules import load_module


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path_factory):
    for key in list(os.environ):
        if key.startswith(SECRET_PREFIXES):
            monkeypatch.delenv(key)
    private = tmp_path_factory.mktemp("private")
    gitconfig = private / "gitconfig"
    gitconfig.write_text(
        "[user]\n\tname = Test\n\temail = test@example.invalid\n"
        "[core]\n\tautocrlf = false\n\tlongpaths = true\n"
        "[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GH_CONFIG_DIR", str(private / "gh"))


@pytest.fixture
def lab_root(tmp_path) -> Path:
    return copy_lab(tmp_path / "lab")


@pytest.fixture
def toy_lab(lab_root, tmp_path) -> Path:
    """A lab with one scaffolded direction whose exp_001 points at fake data."""
    tool("new_direction").new_direction(lab_root, TOY, "Toy paired comparison")
    fake = load_module(lab_root / "directions" / TOY / "tests" / "fake_data.py")
    data = fake.make_dataset(tmp_path / "toy-data", n_subjects=8)
    config = lab_root / "directions" / TOY / "configs" / "exp_001.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "compute": {"est_gpu_hours": 0.5},
                "params": {"data_dir": str(data), "n_subjects": 8, "n_boot": 200, "seed": 1},
            }
        ),
        encoding="utf-8",
    )
    return lab_root
