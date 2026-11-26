"""Keep phenotype data out of results/: reject CSV and JSON files with such column names.

Only NAMES are inspected (CSV/TSV header cells, JSON object keys, listed JSON column names),
never values. A list of names cannot catch everything: a phenotype stored under a neutral name,
or as a value in a long "variable, value" table, passes. Messages quote names only.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

CHECKED_SUFFIXES = (".csv", ".tsv", ".json", ".jsonl")

# A column matches if one of its words is in this set ("patient_age", "AgeAtScan", "mRS_90d").
WORDS = frozenset(
    """
    age ages sex gender race dob bmi mrs tici mtici aspects etiology toast ivt tpa afib sbp dbp
    glucose cholesterol hba1c inr creatinine mortality death deceased diagnosis mrn handedness
    education occupation income zipcode postcode address phone email medication medications
    anticoagulant anticoagulants antiplatelet antiplatelets thrombectomy thrombolysis
    """.split()
)
# ... or if the name with punctuation removed contains one of these.
FRAGMENTS = (
    "nihss", "phenotype", "ethnic", "comorbid", "birth", "hypertension", "diabet", "lipidemia",
    "smoking", "smoker", "alcohol", "atrialfib", "bloodpressure", "heartrate", "bodyweight",
    "bodyheight", "onsetto", "onsettime", "timesinceonset", "pregnan", "patientname", "patientsex",
    "stroketype",
)
# Names written without separators that must match whole: as fragments, "agegroup" would also catch
# "image_group".
WHOLE_NAMES = frozenset(
    "patientage agegroup agecategory ageyears ageatonset ageatscan weightkg heightcm".split()
)


def _words(name: str) -> set[str]:
    spaced = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)  # splits camelCase
    plain = re.findall(r"[a-z]+|\d+", name.lower())  # keeps "mRS" together
    whole = re.findall(r"[a-z0-9]+", name.lower())  # keeps "HbA1c" together
    return {word.lower() for word in (*spaced, *plain, *whole)}


def is_phenotype_name(name: str) -> bool:
    joined = re.sub(r"[^a-z0-9]", "", name.lower())
    if joined in WHOLE_NAMES or _words(name) & WORDS:
        return True
    return any(fragment in joined for fragment in FRAGMENTS)


def _json_names(value: Any):
    """Every object key, plus column names that table-like JSON lists as values."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            if key == "columns" and isinstance(child, list):
                yield from (item for item in child if isinstance(item, str))
            if key == "fields" and isinstance(child, list):
                yield from (str(item["name"]) for item in child if isinstance(item, dict) and "name" in item)
            yield from _json_names(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_names(child)


def names_in(value: Any) -> list[str]:
    """Phenotype-like names in an already parsed JSON value."""
    return sorted({name for name in _json_names(value) if is_phenotype_name(name)})


def _file_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return list(_json_names(json.loads(text)))
    if suffix == ".jsonl":
        return [n for line in text.splitlines() if line.strip() for n in _json_names(json.loads(line))]
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return next(csv.reader(lines[:1], delimiter="\t" if suffix == ".tsv" else ","))


def file_problems(path: Path, label: str | None = None) -> list[str]:
    """Problems with one file; files of other types pass. An unreadable file fails."""
    path, label = Path(path), label or Path(path).name
    if path.suffix.lower() not in CHECKED_SUFFIXES:
        return []
    try:
        found = sorted({name for name in _file_names(path) if is_phenotype_name(name)})
    except (OSError, ValueError, csv.Error, RecursionError) as exc:
        return [f"{label}: cannot be read, so it cannot be checked for phenotype fields "
                f"({type(exc).__name__})"]
    if not found:
        return []
    return [f"{label}: phenotype-like column(s) {', '.join(repr(n[:60]) for n in found[:10])}"]


def tree_problems(folder: Path) -> list[str]:
    folder = Path(folder)
    problems = []
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            problems += file_problems(path, path.relative_to(folder).as_posix())
    return problems
