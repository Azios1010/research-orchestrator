"""The classifiers on frozen features: an L2-regularised multinomial linear probe, and cosine kNN."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def l2_normalise(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, 1e-12)).astype(np.float64)


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    others: list[np.ndarray],
    n_classes: int,
    *,
    c: float,
    class_weight: str | None,
    max_iter: int,
    tol: float,
) -> tuple[list[np.ndarray], int, bool]:
    """Fit on the train set; class probabilities (n, n_classes) for each other set.

    A class that is absent from the train set gets probability 0. Returns also the number of solver
    iterations and whether the solver converged.
    """
    model = LogisticRegression(C=c, class_weight=class_weight, max_iter=max_iter, tol=tol, solver="lbfgs")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if len(model.classes_) < 3:
        raise ValueError("the train set holds fewer than 3 classes")
    out = []
    for x in others:
        full = np.zeros((len(x), n_classes))
        full[:, model.classes_] = model.predict_proba(x)
        out.append(full)
    return out, int(np.max(model.n_iter_)), converged


def knn(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, k: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Majority vote of the k most similar train rows (rows are L2-normalised, so cosine).

    A tie in votes goes to the class with the larger summed similarity. Returns the predicted classes
    and the neighbour indices (n_eval, k) into the train set.
    """
    if len(x_train) < k:
        raise ValueError(f"kNN needs at least {k} train cells, got {len(x_train)}")
    similarity = x_eval @ x_train.T
    neighbours = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    rows = np.arange(len(x_eval))[:, None]
    votes = np.zeros((len(x_eval), n_classes))
    weight = np.zeros((len(x_eval), n_classes))
    np.add.at(votes, (rows, y_train[neighbours]), 1.0)
    np.add.at(weight, (rows, y_train[neighbours]), similarity[rows, neighbours])
    # |summed similarity| <= k, so this term only ever separates classes with equal votes.
    return np.argmax(votes + weight / (4.0 * k), axis=1), neighbours
