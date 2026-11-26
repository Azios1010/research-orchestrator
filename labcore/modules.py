"""Load direction code by path.

Every direction has files called `src/`, `checks.py` and `tests/fake_data.py`, so each one is
imported under a unique module name to keep directions from colliding in one process.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _unique_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"lab_{path.stem}_{digest}"


def _load(path: Path, name: str, package_dir: Path | None) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(package_dir)] if package_dir else None
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_module(path: Path) -> ModuleType:
    """Import one .py file. A fresh copy is executed on every call."""
    path = Path(path)
    return _load(path, _unique_name(path), None)


def load_package(directory: Path) -> str:
    """Import a folder with __init__.py and return the unique package name.

    Submodules then import as f"{name}.data", and relative imports work.
    """
    directory = Path(directory)
    name = _unique_name(directory / "pkg")
    if name not in sys.modules:
        _load(directory / "__init__.py", name, directory)
    return name
