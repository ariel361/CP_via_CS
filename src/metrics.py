"""
metrics.py – evaluation helpers shared across all conformal experiments.

All functions operate on a boolean prediction_sets array of shape (N, C)
and a labels array of shape (N,).
"""

import numpy as np


def marginal_coverage(prediction_sets: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of test samples whose true label is in the prediction set."""
    covered = prediction_sets[np.arange(len(labels)), labels]
    return float(covered.mean())


def class_conditional_coverage(
    prediction_sets: np.ndarray, labels: np.ndarray, num_class: int
) -> np.ndarray:
    """
    Per-class coverage rates.

    Returns
    -------
    cc : ndarray of shape (num_class,)
        cc[y] = fraction of samples with true label y that are covered.
    """
    covered = prediction_sets[np.arange(len(labels)), labels]
    return np.array([
        float(covered[labels == y].mean()) if (labels == y).any() else float("nan")
        for y in range(num_class)
    ])


def worst_class_coverage_deviation(
    cc: np.ndarray, alpha: float
) -> tuple[float, int]:
    """
    Maximum absolute deviation from (1-alpha) across all classes.

    Returns
    -------
    deviation : float
    worst_class : int
    """
    deviations = np.abs(cc - (1 - alpha))
    worst_class = int(np.argmax(deviations))
    return float(deviations[worst_class]), worst_class


def top_k_worst_coverage_deviation(
    cc: np.ndarray, alpha: float, k: int = 5
) -> float:
    """Mean absolute deviation of the k worst-covered classes."""
    worst_indices = np.argsort(cc)[:k]
    return float(np.mean(np.abs(cc[worst_indices] - (1 - alpha))))


def set_size_stats(prediction_sets: np.ndarray) -> dict:
    """Return mean, std, min, and max set sizes."""
    sizes = prediction_sets.sum(axis=1)
    return {
        "mean": float(sizes.mean()),
        "std":  float(sizes.std()),
        "min":  float(sizes.min()),
        "max":  float(sizes.max()),
    }


def mean_distinct_superclasses(
    prediction_sets: np.ndarray, adjacency_matrix: np.ndarray, num_class: int
) -> float:
    """
    Average number of distinct CIFAR-100 superclasses per prediction set.

    Parameters
    ----------
    adjacency_matrix : ndarray, shape (C, C+1)
        Last column stores the integer superclass id for each fine class.
    """
    counts = []
    for i in range(prediction_sets.shape[0]):
        sc = {
            adjacency_matrix[j, num_class]
            for j, v in enumerate(prediction_sets[i]) if v
        }
        counts.append(len(sc))
    return float(np.mean(counts))


def print_metrics(
    coverage: float,
    mean_cc: float,
    worst_dev: float,
    worst_class: int,
    top5_dev: float,
    sizes: dict,
    mean_sc: float,
) -> None:
    """Pretty-print the standard metric block."""
    print(f"Marginal coverage            : {coverage:.4f}")
    print(f"Mean class-conditional cov.  : {mean_cc:.4f}")
    print(f"Worst-class CC deviation     : {worst_dev:.4f}  (class {worst_class})")
    print(f"Top-5 worst CC deviation     : {top5_dev:.4f}")
    print(f"Avg set size                 : {sizes['mean']:.3f}  "
          f"(std {sizes['std']:.3f}, min {sizes['min']:.0f}, max {sizes['max']:.0f})")
    if mean_sc is not None:
        print(f"Avg distinct superclasses    : {mean_sc:.3f}")
