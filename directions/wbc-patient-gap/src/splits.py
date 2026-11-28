"""Who trains and who is evaluated: patient folds, the eval/donor halves, and the two train sets.

All functions work on cell indices and are deterministic for a given generator.
"""

from __future__ import annotations

import numpy as np


def patient_folds(patients: np.ndarray, n_folds: int, seed: int) -> dict[str, int]:
    """patient -> fold, from a seeded order of the sorted patient names. Fold sizes differ by at most 1."""
    names = sorted(set(patients.tolist()))
    if len(names) < n_folds:
        raise ValueError(f"{len(names)} patient(s) cannot fill {n_folds} folds")
    order = np.random.default_rng(seed).permutation(len(names))
    return {names[i]: position % n_folds for position, i in enumerate(order)}


def split_eval_donor(
    cells: np.ndarray, patients: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Halve each test patient's cells into `eval` and `donor`, class by class.

    A class with an odd number of cells gives the extra one to each side in turn, from a random first
    side per patient, so a patient's two halves differ by at most one cell in total.
    """
    halves: tuple[list[int], list[int]] = ([], [])
    for patient in sorted(set(patients[cells].tolist())):
        own = cells[patients[cells] == patient]
        extra_side = int(rng.integers(2))
        for label in np.unique(labels[own]):
            group = rng.permutation(own[labels[own] == label])
            half = len(group) // 2
            sizes = [half, half]
            if len(group) % 2:
                sizes[extra_side] += 1
                extra_side = 1 - extra_side
            halves[0].extend(group[: sizes[0]].tolist())
            halves[1].extend(group[sizes[0] :].tolist())
    return np.sort(np.array(halves[0], dtype=np.int64)), np.sort(np.array(halves[1], dtype=np.int64))


def shared_train(
    disjoint: np.ndarray, donors: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """`disjoint` with, class by class, n of its cells replaced by n donor cells.

    n is the number of donor cells of the class, or the number of train cells of the class if that is
    smaller. So the size and the count of every class stay what they were.
    """
    kept = []
    for label in np.unique(labels[disjoint]):
        own = disjoint[labels[disjoint] == label]
        offered = donors[labels[donors] == label]
        n = min(len(own), len(offered))
        stay = rng.permutation(own)[n:]
        take = rng.permutation(offered)[:n]
        kept.append(np.concatenate([stay, take]))
    return np.sort(np.concatenate(kept))
