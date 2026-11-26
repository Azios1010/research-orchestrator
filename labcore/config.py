"""Load and validate direction.yaml and experiment configs."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from labcore.io import read_yaml
from labcore.privacy import is_phenotype_name
from labcore.schema import schema_errors
from labcore.stats.scorecard import COMMON_FIELDS
from labcore.verdict import gates as gate_parser

DEFAULT_MAX_L1_ROUNDS = 2
EXP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ConfigError(ValueError):
    def __init__(self, source: str, errors: list[str]):
        self.errors = errors
        super().__init__(f"{source}:\n  " + "\n  ".join(errors))


def direction_errors(document: object) -> list[str]:
    """Schema errors plus checks the schema cannot express (gate syntax and gate field names)."""
    errors = schema_errors(document, "direction")
    if errors:
        return errors
    for name in document["scorecard_map"]["metrics"]:
        if is_phenotype_name(name):
            errors.append(f"scorecard_map: {name!r} looks like a phenotype field; results/ may not hold it")
    known = set(COMMON_FIELDS) | set(document["scorecard_map"]["metrics"])
    for name in ("GO", "PARK"):
        try:
            node = gate_parser.parse(document["gates"][name])
        except gate_parser.GateError as exc:
            errors.append(f"gates.{name}: {exc}")
            continue
        for missing in sorted(gate_parser.fields(node) - known):
            errors.append(f"gates.{name}: {missing!r} is not a scorecard field (add it to scorecard_map)")
    return errors


def load_direction(direction_dir: Path) -> dict:
    path = Path(direction_dir) / "direction.yaml"
    if not path.is_file():
        raise ConfigError(str(path), ["file not found"])
    try:
        document = read_yaml(path)
    except yaml.YAMLError as exc:
        raise ConfigError(str(path), [f"invalid YAML: {exc}"]) from exc
    errors = direction_errors(document)
    if errors:
        raise ConfigError(str(path), errors)
    document["compute"].setdefault("max_l1_rounds", DEFAULT_MAX_L1_ROUNDS)
    return document


def experiment_path(direction_dir: Path, exp: str) -> Path:
    if not EXP_NAME.match(exp):
        raise ConfigError(exp, ["experiment names may only contain letters, digits, '_' and '-'"])
    return Path(direction_dir) / "configs" / f"{exp}.yaml"


def load_experiment(path: Path) -> dict:
    """An experiment config is `compute.est_gpu_hours` plus free-form `params`."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(str(path), ["experiment config not found"])
    document = read_yaml(path)
    errors = []
    if not isinstance(document, dict):
        raise ConfigError(str(path), ["must be a YAML mapping"])
    estimate = (document.get("compute") or {}).get("est_gpu_hours")
    if not isinstance(estimate, (int, float)) or isinstance(estimate, bool) or estimate <= 0:
        errors.append("compute.est_gpu_hours must be a positive number")
    if not isinstance(document.get("params"), dict):
        errors.append("params must be a mapping")
    if errors:
        raise ConfigError(str(path), errors)
    return document
