"""Where things live inside a lab checkout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DRYRUN_DIRNAME = ".dryrun"


@dataclass(frozen=True)
class Lab:
    """Paths of one lab checkout. State and results can be redirected (dry runs, CI)."""

    root: Path
    state_dir: Path
    results_dir: Path

    @classmethod
    def at(
        cls, root: Path | str, state_dir: Path | str | None = None, results_dir: Path | str | None = None
    ) -> Lab:
        root = Path(root).resolve()
        return cls(
            root=root,
            state_dir=Path(state_dir).resolve() if state_dir else root / "state",
            results_dir=Path(results_dir).resolve() if results_dir else root / "results",
        )

    @classmethod
    def dry_run(cls, root: Path | str) -> Lab:
        """A lab whose state and results go to .dryrun/, never to the bot-owned folders."""
        root = Path(root).resolve()
        return cls.at(root, root / DRYRUN_DIRNAME / "state", root / DRYRUN_DIRNAME / "results")

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.yaml"

    @property
    def directions_dir(self) -> Path:
        return self.root / "directions"

    def direction_dir(self, slug: str) -> Path:
        return self.directions_dir / slug


def find_root(start: Path | str | None = None) -> Path:
    """Walk upwards from `start` (default: cwd) to the folder that holds registry.yaml."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "registry.yaml").is_file():
            return candidate
    raise FileNotFoundError(f"no registry.yaml found in {here} or any parent folder; pass --root")
