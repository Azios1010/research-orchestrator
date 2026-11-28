"""The design: for every split seed and fold, both arms train a probe and score the same eval cells."""

from __future__ import annotations

import hashlib

import numpy as np

from . import evaluate as ev
from .data import CLASSES
from .probe import fit_predict, knn
from .splits import patient_folds, shared_train, split_eval_donor

N_CLASSES = len(CLASSES)
N_GROUPS = len(ev.GROUP_NAMES)


def set_hash(hashes: np.ndarray) -> str:
    return hashlib.sha256("\n".join(sorted(hashes.tolist())).encode()).hexdigest()[:16]


def run_experiment(
    x: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    hashes: np.ndarray,
    x_external: np.ndarray,
    external_groups: np.ndarray,
    cfg: dict,
    log=print,
) -> dict:
    """`x` and `x_external` are L2-normalised features. Returns counts and bookkeeping, no statistics."""
    names = np.array(sorted(set(patients.tolist())))
    position = np.searchsorted(names, patients)
    fold_of = patient_folds(patients, cfg["n_folds"], cfg["fold_seed"])
    cell_fold = np.array([fold_of[p] for p in patients.tolist()])
    n_seeds, n_patients = len(cfg["split_seeds"]), len(names)

    shape = (n_seeds, n_patients, N_CLASSES, 3)
    probe = {arm: np.zeros(shape, dtype=np.int64) for arm in ev.ARMS}
    neighbour = {arm: np.zeros(shape, dtype=np.int64) for arm in ev.ARMS}
    heldout = np.zeros((n_seeds, N_GROUPS, 3), dtype=np.int64)
    external = np.zeros((n_seeds, N_GROUPS, 3), dtype=np.int64)
    shares = np.zeros(4)
    folds, split_counts = [], []

    for s, seed in enumerate(cfg["split_seeds"]):
        for fold in range(cfg["n_folds"]):
            test = np.flatnonzero(cell_fold == fold)
            disjoint = np.flatnonzero(cell_fold != fold)
            evaluated, donors = split_eval_donor(test, patients, labels, np.random.default_rng([seed, fold]))
            if not len(evaluated):
                raise RuntimeError(f"fold {fold}, seed {seed}: no cell to evaluate")
            train_of = {
                "disjoint": disjoint,
                "shared": shared_train(disjoint, donors, labels, np.random.default_rng([seed, fold, 1])),
            }
            test_patients = set(patients[test].tolist())
            for arm in ev.ARMS:
                train = train_of[arm]
                others = [x[evaluated]] + ([x_external] if arm == "disjoint" else [])
                settings = {name: cfg[f"probe_{name}"] for name in ("c", "class_weight", "max_iter", "tol")}
                probabilities, iterations, converged = fit_predict(
                    x[train], labels[train], others, N_CLASSES, **settings
                )
                predicted = probabilities[0].argmax(axis=1)
                unit, true = position[evaluated], labels[evaluated]
                probe[arm][s] += ev.counts_by_unit(unit, true, predicted, n_patients, N_CLASSES)
                voted, indices = knn(x[train], labels[train], x[evaluated], cfg["knn_k"], N_CLASSES)
                neighbour[arm][s] += ev.counts_by_unit(unit, true, voted, n_patients, N_CLASSES)
                if arm == "shared":
                    shares += ev.neighbour_shares(
                        indices, patients[train], labels[train], patients[evaluated]
                    )
                else:
                    groups = ev.group_of_label(true)
                    inside = groups >= 0
                    heldout[s] += ev.counts_by_unit(
                        np.zeros(inside.sum(), dtype=np.int64), groups[inside],
                        ev.group_prediction(probabilities[0][inside]), 1, N_GROUPS,
                    )[0]
                    external[s] += ev.counts_by_unit(
                        np.zeros(len(external_groups), dtype=np.int64), external_groups,
                        ev.group_prediction(probabilities[1]), 1, N_GROUPS,
                    )[0]
                # What was really used, for checks.py.
                from_test = np.array([p in test_patients for p in patients[train].tolist()])
                folds.append({
                    "fold": fold, "split_seed": seed, "arm": arm,
                    "n_test_patients": len(test_patients), "n_train": len(train),
                    "n_train_donor": int(np.isin(train, donors).sum()), "n_eval": len(evaluated),
                    "eval_set_hash": set_hash(hashes[evaluated]),
                    "n_eval_in_train": int(np.isin(evaluated, train).sum()),
                    "n_eval_hash_in_train": int(np.isin(hashes[evaluated], hashes[train]).sum()),
                    "n_test_patient_cells_in_train": int(from_test.sum()),
                    "probe_iterations": iterations, "probe_converged": converged,
                })
                for k, name in enumerate(CLASSES):
                    split_counts.append({
                        "fold": fold, "split_seed": seed, "arm": arm, "class_name": name,
                        "n_train": int((labels[train] == k).sum()),
                        "n_train_donor": int((labels[train][np.isin(train, donors)] == k).sum()),
                        "n_eval": int((true == k).sum()),
                    })
            log(f"seed {seed} fold {fold}: {len(evaluated)} eval, {len(donors)} donor, {len(disjoint)} train")

    return {
        "patients": names, "probe": probe, "knn": neighbour, "heldout": heldout, "external": external,
        "shares": shares, "folds": folds, "split_counts": split_counts,
    }
