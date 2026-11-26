"""JSON Schema validation against the schemas in labcore/schemas/."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


@cache
def load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as handle:
        return json.load(handle)


def schema_errors(payload: Any, name: str, definition: str | None = None) -> list[str]:
    """Return human-readable errors; an empty list means the payload is valid."""
    schema = load_schema(name)
    if definition:
        schema = {"$ref": f"#/$defs/{definition}", "$defs": schema["$defs"]}
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(payload), key=lambda e: [str(p) for p in e.absolute_path]):
        where = ".".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"{where}: {error.message}")
    return errors
