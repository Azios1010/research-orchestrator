"""Everything that turns predictions into a number: counts per patient, F1, the patient bootstrap,
the seven-group comparison and the neighbour shares.

Primary quantity: macro-F1 over the five neutrophil maturation classes. For one split seed, true and
false positives and false negatives are summed over all evaluated patients, F1 is computed per class and
averaged over the five classes; the value of an arm is the mean over the split seeds. The contrast is
shared minus disjoint, with a percentile bootstrap that resamples patients, the same resample for both
arms and every seed.
"""

from __future__ import annotations

import numpy as np

from .data import CLASSES

ARMS = ("disjoint", "shared")
CONTRAST = "shared-disjoint"
PRIMARY = ("Segmented neutrophils", "Band neutrophils", "Metamyelocyte", "Myelocyte", "Promyelocyte")
PRIMARY_INDEX = tuple(CLASSES.index(name) for name in PRIMARY)
# The seven groups shared with the Acevedo set. "Blast, no lineage spec" belongs to none of them.
GROUPS = {
    "neutrophil": ("Segmented neutrophils", "Band neutrophils"),
    "eosinophil": ("Eosinophils",),
    "basophil": ("Basophil",),
    "lymphocyte": ("Lymphocyte", "Lymphocyte, variant"),
    "monocyte": ("Monocyte",),
    "immature granulocyte": ("Promyelocyte", "Myelocyte", "Metamyelocyte"),
    "erythroblast": ("Erythroblast",),
}
GROUP_NAMES = tuple(GROUPS)
TP, FP, FN = 0, 1, 2


def counts_by_unit(
    unit: np.ndarray, true: np.ndarray, pred: np.ndarray, n_units: int, n_classes: int
) -> np.ndarray:
    """(n_units, n_classes, [tp, fp, fn]). `unit` is the patient position of each cell."""
    out = np.zeros((n_units, n_classes, 3), dtype=np.int64)
    right = true == pred
    np.add.at(out, (unit[right], true[right], TP), 1)
    np.add.at(out, (unit[~right], pred[~right], FP), 1)
    np.add.at(out, (unit[~right], true[~right], FN), 1)
    return out


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """2 tp / (2 tp + fp + fn) over the last axis; 0 for a class with no true and no predicted cell."""
    counts = np.asarray(counts, dtype=float)
    top = 2.0 * counts[..., TP]
    bottom = top + counts[..., FP] + counts[..., FN]
    return np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)


def macro_f1(counts: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    """counts (..., K, 3) -> macro-F1 over `classes`."""
    return f1_from_counts(counts)[..., list(classes)].mean(axis=-1)


def bootstrap_weights(n_units: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_units): how often each patient is drawn in each resample."""
    drawn = rng.integers(0, n_units, size=(n_boot, n_units))
    return np.stack([np.bincount(row, minlength=n_units) for row in drawn])


def paired_bootstrap(
    disjoint: np.ndarray, shared: np.ndarray, classes: tuple[int, ...], weights: np.ndarray
) -> dict:
    """Both arrays are (seeds, patients, K, 3). Returns shared - disjoint with its 95% interval."""
    if disjoint.shape != shared.shape or disjoint.ndim != 4 or weights.shape[1] != disjoint.shape[1]:
        raise ValueError("the arms need equal (seeds, patients, classes, 3) tables matching the weights")

    def point(table: np.ndarray) -> np.ndarray:
        return macro_f1(table.sum(axis=1), classes)  # one value per seed

    def resampled(table: np.ndarray) -> np.ndarray:
        summed = np.tensordot(weights, table, axes=(1, 1))  # (boot, seeds, K, 3)
        return macro_f1(summed, classes).mean(axis=1)

    by_seed = point(shared) - point(disjoint)
    deltas = resampled(shared) - resampled(disjoint)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_patients": int(disjoint.shape[1]),
        "mean_delta": float(by_seed.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "by_seed": [float(v) for v in by_seed],
    }


def arm_f1(table: np.ndarray, classes: tuple[int, ...]) -> float:
    """(seeds, patients, K, 3) -> the arm's value: macro-F1 per seed, averaged over seeds."""
    return float(macro_f1(table.sum(axis=1), classes).mean())


def group_of_label(labels: np.ndarray) -> np.ndarray:
    """Index into GROUP_NAMES for each 12-class label; -1 for a class outside the seven groups."""
    lookup = np.full(len(CLASSES), -1, dtype=np.int64)
    for g, members in enumerate(GROUPS.values()):
        for name in members:
            lookup[CLASSES.index(name)] = g
    return lookup[labels]


def group_prediction(probabilities: np.ndarray) -> np.ndarray:
    """Sum the 12-class probabilities inside each group and take the largest group."""
    summed = np.stack(
        [probabilities[:, [CLASSES.index(m) for m in members]].sum(axis=1) for members in GROUPS.values()],
        axis=1,
    )
    return summed.argmax(axis=1)


def neighbour_shares(
    neighbours: np.ndarray,
    train_patients: np.ndarray,
    train_labels: np.ndarray,
    eval_patients: np.ndarray,
) -> np.ndarray:
    """Summed over eval cells: [observed, random, class-matched random, number of eval cells].

    observed: share of a cell's neighbours that come from its own patient. random: the share of the
    train set that comes from that patient. class-matched random: for each neighbour, the share of the
    train cells of the neighbour's class that come from that patient, averaged over the neighbours.
    """
    names, patient_code = np.unique(np.concatenate([train_patients, eval_patients]), return_inverse=True)
    train_code, eval_code = patient_code[: len(train_patients)], patient_code[len(train_patients) :]
    table = np.zeros((len(names), len(CLASSES)))
    np.add.at(table, (train_code, train_labels), 1.0)
    per_class = table.sum(axis=0)
    observed = (train_code[neighbours] == eval_code[:, None]).mean(axis=1)
    at_random = table.sum(axis=1)[eval_code] / len(train_labels)
    labels = train_labels[neighbours]
    matched = (table[eval_code[:, None], labels] / per_class[labels]).mean(axis=1)
    return np.array([observed.sum(), at_random.sum(), matched.sum(), len(eval_code)])
